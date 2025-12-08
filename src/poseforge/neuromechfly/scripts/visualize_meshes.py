import numpy as np
import pyvista as pv
import h5py
from tqdm import trange
from xml.etree import ElementTree
from pathlib import Path
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from pyvista import PolyData
from loguru import logger

from poseforge.neuromechfly.constants import legs, all_segment_names_per_leg
from poseforge.util.data import bulk_data_dir


def load_neuromechfly_meshes(flygym_path: Path | str) -> dict[str, PolyData]:
    """Load NeuroMechFly meshes from the specified FlyGym installation path."""
    # Define paths
    mesh_dir = Path(flygym_path) / "data/mesh"
    mjcf_path = Path(flygym_path) / "data/mjcf/neuromechfly_seqik_kinorder_ypr.xml"

    # Load NeuroMechFly MJCF file (which is just an XML file)
    mjcf_tree = ElementTree.parse(mjcf_path)

    # Load body segment meshes
    mesh_lookup = {}
    for mesh_element in mjcf_tree.findall("asset/mesh"):  # findall iterates in order
        mesh_path = mesh_dir / Path(mesh_element.attrib["file"]).name
        mesh_name = mesh_element.attrib["name"]
        if not mesh_name.startswith("mesh_"):
            raise RuntimeError(
                "Unexpected mesh specified in MJCF: name doesn't start with 'mesh_'"
            )
        mesh_name = mesh_name[len("mesh_") :]
        mesh_scale_str = mesh_element.attrib["scale"]
        try:
            mesh_scale_xyz = [float(s) for s in mesh_scale_str.split()]
        except ValueError as e:
            raise RuntimeError(
                "Unexpected mesh specified in MJCF: scale not following format "
                "'xscale yscale zscale' (each a number)"
            ) from e

        # Load mesh and apply initial scaling transform
        # (lengths in NeuroMechFly simulations are in mm instead of m - this is how NMF
        # tells MuJoCo to scale everything accordingly)
        mesh: PolyData = pv.read(mesh_path)
        if mesh is None:
            raise RuntimeError(f"Mesh '{mesh_name}' from '{mesh_path}' is empty")
        scale_transform = np.eye(4)
        np.fill_diagonal(scale_transform, [*mesh_scale_xyz, 1])
        mesh = mesh.transform(scale_transform, inplace=False)
        mesh_lookup[mesh_name] = mesh

    return mesh_lookup


@dataclass
class Pose6DSequence:
    # XYZ position relative to camera (n_frames, n_segments, 3)
    pos_ts: np.ndarray
    # Quaternion rotation relative to camera in scalar-first order (wxyz)
    # (n_frames, n_segments, 4)
    quat_ts: np.ndarray
    # Names of body segments included in pos_ts and quat_ts
    segments: list[str]

    @classmethod
    def from_processed_simulation_data(
        cls,
        data_path: Path,
        segments: list[str] | None,
        relative_to_camera: bool = True,
    ) -> "Pose6DSequence":
        """
        Load 6D pose sequence from processed NeuroMechFly simulation data file.

        Args:
            data_path: Path to processed simulation data HDF5 file
            segments: List of segment names to load. If None, load all recorded segments
            relative_to_camera: If True, load segment poses relative to camera.
                If False, load raw global segment poses (pre-postprocessing).

        Returns:
            An instance of Pose6DSequence containing the loaded data
        """
        pos_ts, quat_ts, segments = _load_simulation_data(
            data_path, segments, load_raw=not relative_to_camera
        )
        return cls(pos_ts=pos_ts, quat_ts=quat_ts, segments=segments)

    def __len__(self):
        return self.pos_ts.shape[0]

    def render(
        self,
        mesh_assets: dict[str, PolyData],
        render_fps: int,
        output_path: Path | str | None = None,
        display_live: bool = False,
        theme: str = "dark",
        disable_pbar: bool = False,
    ) -> None:
        """Render the 6D pose sequence using NeuroMechFly meshes in PyVista.

        Args:
            mesh_assets: Dictionary mapping segment names to PyVista PolyData meshes.
            render_fps: Frames per second for rendering.
            output_path: If display_live is False, path to save the rendered video.
            display_live: If True, display the rendering live instead of saving to file.
            theme: Display theme, either "dark" or "light".
            disable_pbar: If True, disable the progress bar when saving to file.
        """
        # Filter & check mesh files from NeuroMechFly
        mesh_assets = {k: v for k, v in mesh_assets.items() if k in self.segments}
        missing_keys = set(self.segments) - set(mesh_assets.keys())
        if missing_keys:
            logger.critical(
                "The following segments from Pose6DSequence are not found in "
                f"NeuroMechFly meshes: {missing_keys}"
            )
            raise KeyError("Some meshes are missing")

        # Set up rendering
        plotter, plotted_meshes = _set_up_renderer(mesh_assets, display_live, theme)

        def _update_scene_by_frameid(frameid):
            _transform_all_meshes(
                mesh_assets,
                plotted_meshes,
                self.pos_ts[frameid, :, :],
                self.quat_ts[frameid, :, :],
                self.segments,
            )

        # Set up first frame
        _update_scene_by_frameid(0)
        plotter.reset_camera()

        # Render each frame
        if display_live:
            if output_path is not None:
                logger.warning(
                    "Output path is specified but display_live is set to true. "
                    "Nothing will be saved and the output path will be ignored."
                )

            plotter.add_timer_event(
                max_steps=len(self),
                duration=int(1000 / render_fps),  # in ms
                callback=_update_scene_by_frameid,
            )
            plotter.show()
        else:
            if output_path is None:
                logger.critical(
                    "When Pose6DSequenceRenderer.display_live is set to false, "
                    "output_path must be specified."
                )
                raise ValueError("Video output path not specified.")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plotter.open_movie(output_path, framerate=render_fps)
            for i in trange(len(self), disable=disable_pbar):
                _update_scene_by_frameid(i)
                plotter.write_frame()
            plotter.close()


def _load_simulation_data(
    processed_data_path: Path | str, segments: list[str] | None, load_raw: bool = False
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load 6D pose data from processed NeuroMechFly simulation data file.

    Args:
        processed_data_path: Path to processed simulation data HDF5 file.
        segments: List of segment names to load. If None, load all recorded segments.
        load_raw: If True, load raw global segment poses (pre-postprocessing). If False,
            load postprocessed segment poses relative to camera.

    Returns:
        pos_ts: Array of shape (n_frames, n_segments, 3) with XYZ positions.
        quat_ts: Array of shape (n_frames, n_segments, 4) with quaternion rotations
            in scalar-first (wxyz) order.
        segments: List of segment names corresponding to the loaded data.
    """
    with h5py.File(processed_data_path, "r") as f:
        if load_raw:
            grp = f["raw/body_segment_states/"]
            all_recorded_segments = list(grp.attrs["keys"])
            pos_ts = grp["pos_global"][:]
            quat_ts = grp["quat_global"][:]
        else:
            grp = f["postprocessed/mesh_pose6d_rel_camera"]  # relative to camera
            all_recorded_segments = list(grp.attrs["keys"])
            # grp["pos_rel_cam"][:, seg_indices, :] would be better, but h5py only
            # supports index-based fancy indexing if the indices are monotonic, so meh
            pos_ts = grp["pos_rel_cam"][:]
            quat_ts = grp["quat_rel_cam"][:]
        if segments is None:
            segments = all_recorded_segments
        else:
            seg2idx = {seg: i for i, seg in enumerate(all_recorded_segments)}
            try:
                seg_indices = np.array([seg2idx[seg] for seg in segments])
            except KeyError as e:
                raise KeyError(f"Some segments not found in simulation data") from e
            pos_ts = pos_ts[:, seg_indices, :]
            quat_ts = quat_ts[:, seg_indices, :]

    n_frames, n_segments, _ = pos_ts.shape
    assert pos_ts.shape == (n_frames, n_segments, 3), "Invalid shape for mesh pos_ts"
    assert quat_ts.shape == (n_frames, n_segments, 4), "Invalid shape for mesh quat_ts"
    return pos_ts, quat_ts, segments


def _set_up_renderer(
    nmf_meshes: dict[str, PolyData], display_live: bool = False, theme: str = "dark"
) -> tuple[pv.Plotter, dict[str, PolyData]]:
    """Set up PyVista plotter and add NeuroMechFly meshes to it"""
    # Set up plotter
    plotter = pv.Plotter(off_screen=not display_live)
    plotter.show_axes()

    # Apply display theme
    if theme.lower() == "dark":
        plotter.set_background("black")
        mesh_color = "#eeeeee"
    elif theme.lower() == "light":
        plotter.set_background("white")
        mesh_color = "#bbbbbb"
    else:
        raise ValueError(f"Undefined display theme '{theme}'")

    # Add meshes initially (don't care about positions)
    plotted_meshes = {}
    for seg_name, mesh in nmf_meshes.items():
        mesh = nmf_meshes[seg_name].copy()
        plotter.add_mesh(mesh, show_edges=False, name=seg_name, color=mesh_color)
        plotted_meshes[seg_name] = mesh

    # Set up camera
    plotter.camera.position = (0, 0, 0)
    plotter.camera.focal_point = (0, 0, 1)  # look down +Z axis
    plotter.camera.up = (0, -1, 0)  # -Y is up (OpenCV/MuJoCo convention)

    return plotter, plotted_meshes


def _transform_one_mesh(mesh: PolyData, pos: np.ndarray, quat: np.ndarray) -> PolyData:
    """Apply 6D pose (translation + quaternion rotation) to a mesh"""
    rot = Rotation.from_quat(quat, scalar_first=True)  # MuJoCo uses wxyz order
    transform = np.eye(4)
    transform[:3, :3] = rot.as_matrix()
    transform[:3, 3] = pos
    return mesh.transform(transform, inplace=False)


def _transform_all_meshes(
    mesh_assets: dict[str, PolyData],
    plotted_meshes: dict[str, PolyData],
    pos_all_segments: np.ndarray,
    quat_all_segments: np.ndarray,
    segment_names: list[str],
) -> None:
    """Update all meshes to the specified 6D poses"""
    assert pos_all_segments.shape == (len(segment_names), 3)
    assert quat_all_segments.shape == (len(segment_names), 4)
    for i, seg_name in enumerate(segment_names):
        transformed_mesh = _transform_one_mesh(
            mesh_assets[seg_name], pos_all_segments[i, :], quat_all_segments[i, :]
        )
        plotted_meshes[seg_name].points = transformed_mesh.points


if __name__ == "__main__":
    flygym_dir = Path("~/projects/flygym/flygym").expanduser()
    sample_sim_path = (
        bulk_data_dir
        / "nmf_rendering/BO_Gal4_fly1_trial001/segment_000/subsegment_001/processed_simulation_data.h5"
    )
    replayed_segments = [
        f"{leg}{seg}" for leg in legs for seg in all_segment_names_per_leg
    ] + ["Thorax"]

    nmf_mesh_assets = load_neuromechfly_meshes(flygym_dir)
    sim_data = Pose6DSequence.from_processed_simulation_data(
        sample_sim_path, segments=replayed_segments, relative_to_camera=True
    )
    sim_data.render(
        mesh_assets=nmf_mesh_assets,
        render_fps=33,
        output_path=sample_sim_path.parent / "pose6d_render.mp4",
        display_live=True,
        theme="light",
    )
