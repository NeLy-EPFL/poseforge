import matplotlib
from poseforge.util.plot import configure_matplotlib_style

matplotlib.use("Agg")
configure_matplotlib_style()

import torch
import numpy as np
import torchvision.transforms as transforms
import h5py
import matplotlib.pyplot as plt
from collections import defaultdict
from time import perf_counter
from tqdm import tqdm
from pathlib import Path
from typing import Any
from fractions import Fraction
from loguru import logger

from pvio.torch_tools import SimpleVideoCollectionLoader
from parallel_animate import Animator, IndexedFrameParams
from parallel_animate.util import get_rendered_frame_ids

import poseforge.pose.keypoints3d as keypoints3d
from poseforge.pose.camera import CameraToWorldMapper
from poseforge.neuromechfly.constants import (
    keypoint_segments_canonical,
    legs,
    leg_keypoints_canonical,
    kchain_plotting_colors,
)


def predict_keypoints3d(
    *,
    keypoints3d_output_path: Path,
    aligned_behavior_video_path: Path,
    keypoints3d_model_config: dict[str, Any],
    device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
    camera_pos: tuple[float, float, float] = (0.0, 0.0, -100.0),
    camera_fov_deg: float = 5.0,
    camera_rendering_size: tuple[int, int] = (464, 464),
    camera_rotation_euler: tuple[float, float, float] = (0, np.pi, -np.pi / 2),
    loading_batch_size: int = 128,
    loading_n_workers: int = 8,
    loading_buffer_size: int = 128,
    loading_cache_video_metadata: bool = True,
) -> None:
    logger.info("Estimating 3D keypoint positions for behavior video")

    # Set up model and pipeline
    architecture_config_path = Path(
        keypoints3d_model_config["architecture_config"]
    ).expanduser()
    logger.info(
        f"Setting up 3D keypoints model from architecture config "
        f"{architecture_config_path}"
    )
    model = keypoints3d.Pose2p5DModel.create_architecture_from_config(
        architecture_config_path
    ).to(device)
    ckpt_path = Path(keypoints3d_model_config["checkpoint"]).expanduser()
    logger.info(f"Loading 3D keypoints model weights from {ckpt_path}")
    weights_config = keypoints3d.config.ModelWeightsConfig(model_weights=ckpt_path)
    model.load_weights_from_config(weights_config)
    logger.info("Creating 3D keypoints inference pipeline")
    pipeline = keypoints3d.Pose2p5DPipeline(model, device=device, use_float16=True)

    # Set up camera mapper
    cam_mapper = CameraToWorldMapper(
        camera_pos, camera_fov_deg, camera_rendering_size, camera_rotation_euler
    )

    # Create video loader
    logger.info("Creating video loader for 3D keypoints prediction")
    working_size = keypoints3d_model_config["working_size"]
    video_loader = SimpleVideoCollectionLoader(
        [aligned_behavior_video_path],
        transform=transforms.Resize((working_size, working_size)),
        batch_size=loading_batch_size,
        num_workers=loading_n_workers,
        buffer_size=loading_buffer_size,
        use_cached_video_metadata=loading_cache_video_metadata,
    )

    # Create output file
    logger.info(
        f"Creating H5 file for 3D keypoint position predictions: "
        f"{keypoints3d_output_path}"
    )
    n_frames_total = len(video_loader.dataset)
    with h5py.File(keypoints3d_output_path, "w") as f:
        keypoint_names = keypoints3d_model_config["keypoint_names"]
        assert keypoint_names == keypoint_segments_canonical, (
            "Keypoint names in model config do not match expected canonical "
            "keypoint names from "
            "poseforge.neuromechfly.constants.keypoint_segments_canonical"
        )
        f.attrs["keypoint_names"] = keypoint_names
        n_keypoints = len(keypoint_names)

        def _create_dataset(name, shape):
            return f.create_dataset(
                name, shape=shape, dtype="float32", compression="gzip"
            )

        ds_pred_xy = _create_dataset("pred_xy", (n_frames_total, n_keypoints, 2))
        ds_pred_depth = _create_dataset("pred_depth", (n_frames_total, n_keypoints))
        ds_conf_xy = _create_dataset("conf_xy", (n_frames_total, n_keypoints))
        ds_conf_depth = _create_dataset("conf_depth", (n_frames_total, n_keypoints))
        ds_pred_world_xyz = _create_dataset(
            "pred_world_xyz", (n_frames_total, n_keypoints, 3)
        )

        # Run inference
        logger.info("Running inference for 3D keypoints prediction")
        for batch in tqdm(
            video_loader, desc="Predicting keypoints3d", unit="batch", disable=None
        ):
            # Forward pass
            # No need to move data to GPU and back, Pipeline.inference handles that
            frames = batch["frames"]
            frame_ids = batch["frame_indices"]
            assert (np.array(batch["video_indices"]) == 0).all()
            pred_dict = pipeline.inference(frames)
            batch_pred_xy = pred_dict["pred_xy"].numpy()  # (B, n_keypoints, 2)
            batch_pred_depth = pred_dict["pred_depth"].numpy()  # (B, n_keypoints)
            batch_conf_xy = pred_dict["conf_xy"].numpy()  # (B, n_keypoints)
            batch_conf_depth = pred_dict["conf_depth"].numpy()  # (B, n_keypoints)

            # So far, xy are pixel coords and depths are distances from camera in mm
            # By nature of alignment step in spotlight_tools.postprocessing, the
            # camera is "fixed" relative to the fly. Using the camera intrinsics and
            # extrinsics provided in the arguments, map predictions to xyz world
            # coords in mm. Shape should be (B, n_keypoints, 3)
            batch_pred_world_xyz = cam_mapper(batch_pred_xy, batch_pred_depth)

            # Save to H5 file
            start_time = perf_counter()
            ds_pred_xy[frame_ids, ...] = batch_pred_xy
            ds_pred_depth[frame_ids, ...] = batch_pred_depth
            ds_conf_xy[frame_ids, ...] = batch_conf_xy
            ds_conf_depth[frame_ids, ...] = batch_conf_depth
            ds_pred_world_xyz[frame_ids, ...] = batch_pred_world_xyz
            elapsed = perf_counter() - start_time
            logger.debug(f"Saved output for {len(frame_ids)} frames in {elapsed:.3f}s")

    logger.info("3D keypoints prediction complete")


def visualize_keypoints3d(
    *,
    visualization_output_path: Path,
    keypoints3d_output_path: Path,
    invkin_output_path: Path,
    aligned_behavior_video_path: Path,
    recording_fps: Fraction | int,
    play_speed: float,
    rendered_fps: int,
    plotted_image_size: int = 256,
    loading_batch_size: int = 128,
    loading_n_workers: int = 4,
    loading_buffer_size: int = 128,
    loading_cache_video_metadata: bool = True,
    rendering_n_workers: int = 12,
):
    logger.info("Visualizing 3D keypoint position predictions")

    # Create video loader
    logger.info("Creating video loader for 3D keypoint position")
    image_shape = (plotted_image_size, plotted_image_size)
    video_loader = SimpleVideoCollectionLoader(
        [aligned_behavior_video_path],
        transform=transforms.Resize(image_shape),
        batch_size=loading_batch_size,
        num_workers=loading_n_workers,
        buffer_size=loading_buffer_size,
        use_cached_video_metadata=loading_cache_video_metadata,
    )

    # Open bodyseg predictions
    logger.info(f"Loading keypoints3d predictions from {keypoints3d_output_path}")
    with h5py.File(keypoints3d_output_path, "r") as f:
        pred_cam_xy_all = f["pred_xy"][:]
        pred_world_xyz_all = f["pred_world_xyz"][:]
        keypoint_names = list(f.attrs["keypoint_names"])

    if invkin_output_path is not None:
        logger.info(f"Loading inverse kinematics results from {invkin_output_path}")
        with h5py.File(invkin_output_path, "r") as f_inv:
            pass
    else:
        logger.warning("No inverse kinematics output path provided; skipping IK data")

    # Define animator
    animator = _Keypoints3DAnimator(
        image_shape=image_shape, with_invkin=invkin_output_path is not None
    )

    # Define frame params iterator and make video
    rendered_ids = get_rendered_frame_ids(
        data_fps=recording_fps,
        play_speed=play_speed,
        rendered_fps=rendered_fps,
        n_data_frames=len(video_loader.dataset),
    )

    def iter_frames():
        for batch in video_loader:
            frames = batch["frames"]  # (B, 1, H, W)
            frame_ids = batch["frame_indices"]
            for i in range(frames.shape[0]):
                frame_id = frame_ids[i]
                if frame_id not in rendered_ids:
                    continue

                # Get input image from video loader
                frame = frames[i, 0, :, :].numpy()
                # Get camera xy predictions
                pred_cam_xy = _group_data_by_kinematic_chain(
                    pred_cam_xy_all[frame_id, ...], keypoint_names=keypoint_names
                )
                # Get 3D world xyz predictions
                raw_pred_world_xyz = _group_data_by_kinematic_chain(
                    pred_world_xyz_all[frame_id, ...], keypoint_names=keypoint_names
                )
                # Get inverse kinematics -> forward kinematics 3D world xyz
                if invkin_output_path is None:
                    fwd_kin_world_xyz = None
                else:
                    fwd_kin_world_xyz = NotImplemented  # TODO

                yield IndexedFrameParams(
                    frame_id=frame_id,
                    params=(frame, pred_cam_xy, raw_pred_world_xyz, fwd_kin_world_xyz),
                )

    animator.make_video(
        visualization_output_path,
        iter_frames(),
        n_frames=int(len(rendered_ids)),
        fps=Fraction(rendered_fps).limit_denominator(100),
        num_workers=rendering_n_workers,
    )


def _group_data_by_kinematic_chain(
    data_block: np.ndarray, keypoint_names: list[str]
) -> dict[str, np.ndarray]:
    expected_keypoint_names = set(keypoint_segments_canonical)
    assert set(keypoint_names) == expected_keypoint_names, (
        "Keypoint names do not match expected canonical keypoint names from "
        "poseforge.neuromechfly.constants.keypoint_segments_canonical"
    )

    data_by_kinematic_chain = defaultdict(list)

    for leg in legs:
        for keypoint in leg_keypoints_canonical:
            keypoint_name = f"{leg}{keypoint}"
            idx = keypoint_names.index(keypoint_name)
            data_by_kinematic_chain[leg].append(data_block[idx, ...])

    for side in "LR":
        keypoint_name = f"{side}Pedicel"
        idx = keypoint_names.index(keypoint_name)
        data_by_kinematic_chain[f"{side}Antenna"].append(data_block[idx, ...])

    return {k: np.array(v) for k, v in data_by_kinematic_chain.items()}


class _Keypoints3DAnimator(Animator):
    raw_pred_line_width = 3
    fwd_kin_line_width = 2
    antenna_marker_size = 5
    x_lim = (0.5, 3.5)
    y_lim = (-3.5, -0.5)
    z_lim = (-0.5, 2.0)
    ax3d_pan_period_in_frames = 300
    ax3d_pan_amplitude = 30
    ax_3d_elevation = 30  # deg

    def __init__(self, *, image_shape: tuple[int, int], with_invkin: bool) -> None:
        self.image_shape = image_shape
        self.with_invkin = with_invkin

    def setup(self):
        fig, (ax_img, _ax_3d_placeholder) = plt.subplots(1, 2, figsize=(8, 4))
        self.artists = {}

        # Set up 2D image axis
        dummy_input_img = np.zeros(self.image_shape)
        dummy_2dpos = {leg: np.zeros((len(leg_keypoints_canonical), 2)) for leg in legs}
        for side in "LR":
            dummy_2dpos[f"{side}Antenna"] = np.zeros((1, 2))
        ax_img.set_title("Recording")
        ax_img.axis("off")
        self.artists["input_img"] = ax_img.imshow(
            dummy_input_img, vmin=0, vmax=1, cmap="gray", origin="upper"
        )
        for kchain_name, xy_seq in dummy_2dpos.items():
            color = kchain_plotting_colors[kchain_name]
            if kchain_name.endswith("Antenna"):
                (artist,) = ax_img.plot(
                    xy_seq[:, 0],
                    xy_seq[:, 1],
                    marker="o",
                    markersize=self.antenna_marker_size,
                    markerfacecolor=color,
                    markeredgecolor=color,
                )
            else:  # legs
                (artist,) = ax_img.plot(
                    xy_seq[:, 0],
                    xy_seq[:, 1],
                    linewidth=self.raw_pred_line_width,
                    color=color,
                )
            self.artists[f"pred_xy_{kchain_name}"] = artist

        # Set up 3D keypoints axis
        dummy_3dpos = {leg: np.zeros((len(leg_keypoints_canonical), 3)) for leg in legs}
        for side in "LR":
            dummy_3dpos[f"{side}Antenna"] = np.zeros((1, 3))
        _ax_3d_placeholder.remove()
        ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
        self.ax_3d = ax_3d  # for updating view later
        ax_3d.set_title("3D Skeleton")
        ax_3d.set_xlim(*self.x_lim)
        ax_3d.set_ylim(*self.y_lim)
        ax_3d.set_zlim(*self.z_lim)
        ax_3d.set_aspect("equal")
        x_range = self.x_lim[1] - self.x_lim[0]
        y_range = self.y_lim[1] - self.y_lim[0]
        z_range = self.z_lim[1] - self.z_lim[0]
        ax_3d.set_box_aspect((x_range, y_range, z_range))

        # 3D keypoints axis: raw model predictions
        for kchain_name, world_xyz_seq in dummy_3dpos.items():
            color = kchain_plotting_colors[kchain_name]
            if kchain_name.endswith("Antenna"):
                (artist,) = ax_3d.plot(
                    world_xyz_seq[:, 0],
                    world_xyz_seq[:, 1],
                    world_xyz_seq[:, 2],
                    marker="o",
                    markersize=self.antenna_marker_size,
                    markerfacecolor=color,
                    markeredgecolor=color,
                )
            else:  # legs
                (artist,) = ax_3d.plot(
                    world_xyz_seq[:, 0],
                    world_xyz_seq[:, 1],
                    world_xyz_seq[:, 2],
                    linewidth=self.raw_pred_line_width,
                    color=color,
                )
            self.artists[f"raw_pred_xyz_{kchain_name}"] = artist

        # 3D keypoints axis: inverse kinematics -> forward kinematics
        for kchain_name, world_xyz_seq in dummy_3dpos.items():
            if not kchain_name.endswith("Antenna"):
                (artist,) = ax_3d.plot(
                    world_xyz_seq[:, 0],
                    world_xyz_seq[:, 1],
                    world_xyz_seq[:, 2],
                    linestyle="-",
                    linewidth=self.fwd_kin_line_width,
                    color="black",
                )
                self.artists[f"fwd_kin_xyz_{kchain_name}"] = artist

        return fig

    def update(self, frame_id: int, data: Any):
        input_image, pred_cam_xy, raw_pred_world_xyz, fwd_kin_world_xyz = data

        # Input image
        self.artists["input_img"].set_data(input_image)

        # Update 3D plot camera view
        azimuth = (
            np.cos(2 * np.pi * frame_id / self.ax3d_pan_period_in_frames)
            * self.ax3d_pan_amplitude
        )
        self.ax_3d.view_init(elev=self.ax_3d_elevation, azim=azimuth)

        for kchain_name in legs + [f"{side}Antenna" for side in "LR"]:
            # 2D keypoints
            artist_cam_xy = self.artists[f"pred_xy_{kchain_name}"]
            artist_cam_xy.set_data(
                pred_cam_xy[kchain_name][:, 0], pred_cam_xy[kchain_name][:, 1]
            )

            # 3D keypoints - raw model predictions
            artist_raw_xyz = self.artists[f"raw_pred_xyz_{kchain_name}"]
            artist_raw_xyz.set_data_3d(
                raw_pred_world_xyz[kchain_name][:, 0],
                raw_pred_world_xyz[kchain_name][:, 1],
                raw_pred_world_xyz[kchain_name][:, 2],
            )

            # 3D keypoints - inverse kinematics -> forward kinematics
            if self.with_invkin and not kchain_name.endswith("Antenna"):
                artist_fwd_xyz = self.artists[f"fwd_kin_xyz_{kchain_name}"]
                artist_fwd_xyz.set_data_3d(
                    fwd_kin_world_xyz[kchain_name][:, 0],
                    fwd_kin_world_xyz[kchain_name][:, 1],
                    fwd_kin_world_xyz[kchain_name][:, 2],
                )
