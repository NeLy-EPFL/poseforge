from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import imageio.v2 as imageio
import shutil
import tempfile
import os
import h5py
from collections import defaultdict
from pathlib import Path

from numpy import ndarray, dtype
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.linalg import rq
from scipy.spatial.transform import Rotation

import poseforge.neuromechfly.constants as constants
from poseforge.util.plot import (
    configure_matplotlib_style,
    get_segmentation_color_palette,
)

configure_matplotlib_style()


class SegmentLabelParser:
    def __init__(self):
        nmf_rendered_colors = {
            "red": [255, 0, 0],
            "green": [0, 255, 0],
            "blue": [0, 0, 255],
            "yellow": [255, 255, 0],
            "magenta": [255, 0, 255],
            "cyan": [0, 255, 255],
            "black": [60, 60, 60],
            "white": [255, 255, 255],
            "gray": [150, 150, 150],
        }
        leg_segments = ["Coxa", "Femur", "Tibia", "Tarsus"]

        self.label_keys = []
        self.label_colors_6d = []

        # Special case: Background
        self.label_keys.append("Background")
        color_6d = np.array(nmf_rendered_colors["black"] + nmf_rendered_colors["black"])
        self.label_colors_6d.append(color_6d)

        # Special case: OtherSegments
        self.label_keys.append("OtherSegments")
        color_6d = np.array(nmf_rendered_colors["gray"] + nmf_rendered_colors["black"])
        self.label_colors_6d.append(color_6d)

        # Special case: Thorax
        self.label_keys.append("Thorax")
        color_6d = np.array(nmf_rendered_colors["gray"] + nmf_rendered_colors["white"])
        self.label_colors_6d.append(color_6d)

        # Legs
        for side in "LR":
            for pos in "FMH":
                for link in leg_segments:
                    leg = f"{side}{pos}"
                    color0 = nmf_rendered_colors[constants.color_by_link[link]]
                    color1 = nmf_rendered_colors[
                        constants.color_by_kinematic_chain[leg]
                    ]
                    color_6d = np.array(list(color0) + list(color1))
                    label = f"{leg}{link}"
                    self.label_keys.append(label)
                    self.label_colors_6d.append(color_6d)

        # Antennas
        for side in "LR":
            color0 = nmf_rendered_colors[constants.color_by_link["Antenna"]]
            color1 = nmf_rendered_colors[constants.color_by_kinematic_chain[side]]
            color_6d = np.array(list(color0) + list(color1))
            label = f"{side}Antenna"
            self.label_keys.append(label)
            self.label_colors_6d.append(color_6d)

        self.label_colors_6d = np.array(self.label_colors_6d)

    def __call__(
        self, images_by_color_coding: list[np.ndarray] | np.ndarray
    ) -> np.ndarray:
        assert (
            len(images_by_color_coding) == 2
        ), "Expecting two images each with a different color coding."
        assert (
            images_by_color_coding[0].shape == images_by_color_coding[1].shape
        ), "Color coding images must have the same shape."
        if not isinstance(images_by_color_coding, list):
            images_by_color_coding = [
                images_by_color_coding[0, ...],
                images_by_color_coding[1, ...],
            ]

        image_6d = np.concatenate(images_by_color_coding, axis=-1)
        sq_distances = np.sum(
            (image_6d[:, :, None, :] - self.label_colors_6d[None, None, :, :]) ** 2,
            axis=-1,
        )
        label_indices = np.argmin(sq_distances, axis=-1)

        return label_indices


def load_video_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    """Load video frames from a video file."""
    if not video_path.is_file():
        raise FileNotFoundError(f"{video_path} is not a file.")

    # Use imageio to read video
    reader = imageio.get_reader(str(video_path))
    frames = []
    num_frames = reader.count_frames()
    for frame in tqdm(reader, total=num_frames, desc="Loading frames", disable=None):
        frames.append(frame)

    # Get FPS from metadata
    fps = reader.get_meta_data().get("fps", 30)  # default to 30 if not available
    reader.close()

    return frames, fps


def get_rotation_angle_and_matrix(forward_vector: np.ndarray) -> tuple[float, ndarray]:
    """Get a rotation matrix that rotates the forward vector to the z-axis."""
    orientation = np.arctan2(forward_vector[1], forward_vector[0])
    rotation_matrix = np.array(
        [
            [np.cos(-orientation), -np.sin(-orientation), 0],
            [np.sin(-orientation), np.cos(-orientation), 0],
            [0, 0, 1],
        ]
    )
    return -orientation, rotation_matrix


def rotate_image(image: np.ndarray, rotation_angle) -> np.ndarray:
    return ndimage.rotate(
        image, np.rad2deg(rotation_angle), reshape=False, order=1, mode="reflect"
    )


def center_square_crop_image(
    image: np.ndarray, side_length: int
) -> tuple[np.ndarray, int, int]:
    """Crop the image to a square of given side length, centered on the image."""
    height, width = image.shape[:2]
    start_col = (width - side_length) // 2
    start_row = (height - side_length) // 2
    cropped_image = image[
        start_row : start_row + side_length, start_col : start_col + side_length
    ]
    return cropped_image, start_col, start_row


def extract_body_segment_positions(
    h5_file: h5py.File,
    frame_idx: int,
    sensor_type: str,
    body_segments: list[str] | None,
):
    # Compile list of body segments
    if body_segments is None:
        body_segments = h5_file["body_segment_states"].attrs["keys"].tolist()

    # Get positions of each keypoint
    pos_world = np.full((3, len(body_segments)), np.nan, dtype=np.float32)
    seg_states_ds = h5_file[f"body_segment_states/{sensor_type}"]
    keys_in_h5_ds = h5_file["body_segment_states"].attrs["keys"].tolist()
    for i, segment_name in enumerate(body_segments):
        idx_in_h5_ds = keys_in_h5_ds.index(segment_name)
        pos_world[:, i] = seg_states_ds[frame_idx, idx_in_h5_ds, :]
    assert not np.any(
        np.isnan(pos_world)
    ), "Some body segment positions are not populated correctly."
    # pos_world_homogeneous: (4, num_keypoints) (last row is all ones)
    pos_world_homogeneous = np.vstack(
        [pos_world, np.ones((1, len(body_segments)), dtype=np.float32)]
    )

    # Project to camera coordinates using the working approach from the snippet
    camera_matrix = h5_file["camera_matrix"][frame_idx, :, :]  # This is 3x4 from MuJoCo

    # Apply camera matrix directly (this gives us the projected coordinates)
    pos_camera = camera_matrix @ pos_world_homogeneous  # (3, num_keypoints)

    # Now that the camera coordinates are computed, we can center the world
    # coordinates. We don't do this before computing camera coordinates because
    # the camera matrix already includes translation information, and we don't
    # want to double-correct it.
    fly_base_pos = h5_file["fly_base_pos"][frame_idx, :]
    pos_world = pos_world - fly_base_pos[:, None]

    # Return keypoint positions as a dictionary
    pos_world_dict = {}
    pos_camera_dict = {}
    for i, segment in enumerate(body_segments):
        pos_world_dict[segment] = pos_world[:, i]
        pos_camera_dict[segment] = pos_camera[:, i]
    return pos_world_dict, pos_camera_dict


def rotate_keypoint_positions_camera(
    keypoints_pos_lookup: dict[str, np.ndarray],
    rotation_matrix: np.ndarray,
    *,
    image_shape: tuple[int, int],
    start_col: int,
    start_row: int,
) -> dict[str, np.ndarray]:
    """
    Apply rotation and cropping transformations to camera/image coordinates.
    Uses a simplified approach based on the working code snippet.

    Args:
        keypoints_pos_lookup: Dict mapping segment names to (x, y, depth) positions
        rotation_matrix: 3x3 rotation matrix used for coordinate transformation
        image_shape: (height, width) of the original image before rotation
        start_col: Cropping offsets after rotation (column)
        start_row: Cropping offsets after rotation (row)

    Returns:
        keypoints_pos_lookup_rotated: Dict mapping segment names to transformed
            (x, y, depth) positions
    """
    keypoints_pos_lookup_rotated = {}
    height, width = image_shape

    for body_segment_name, pos_before_rotation in keypoints_pos_lookup.items():
        # Extract 2D position and depth
        col_row_vec = pos_before_rotation[:2]
        depth = pos_before_rotation[2]
        col_row_vec /= depth  # Normalize by depth to get image coords
        col_row_vec -= np.array([width / 2, height / 2])  # center the coordinates
        col_rot, row_rot = rotation_matrix[:2, :2] @ col_row_vec
        col_rot = col_rot + (width / 2) - start_col
        row_rot = row_rot + (height / 2) - start_row

        # Reconstruct position with original depth
        pos_after_rotation = np.array([col_rot, row_rot, depth], dtype=np.float32)
        keypoints_pos_lookup_rotated[body_segment_name] = pos_after_rotation

    return keypoints_pos_lookup_rotated


def rotate_keypoint_positions_world(
    keypoints_pos_lookup: dict[str, np.ndarray], rotation_matrix: np.ndarray
) -> dict[str, np.ndarray]:
    keypoints_pos_lookup_rotated = {}
    for body_segment_name, pos_before_rotation in keypoints_pos_lookup.items():
        # Apply the rotation matrix to the keypoints
        pos_after_rotation = rotation_matrix @ pos_before_rotation
        keypoints_pos_lookup_rotated[body_segment_name] = pos_after_rotation

    return keypoints_pos_lookup_rotated


def process_single_frame(
    rendered_images: np.ndarray,
    h5_file_path: Path,
    frame_idx: int,
    crop_size: int,
    segment_label_parser: SegmentLabelParser,
):
    # Open the h5 file within the worker process
    with h5py.File(h5_file_path, "r") as h5_file:
        # Get rotation angle and rotation matrix
        forward_vector = h5_file["cardinal_vectors/forward"][frame_idx, :]
        rotation_angle, rotation_matrix = get_rotation_angle_and_matrix(forward_vector)

        # Get image shape before rotation for keypoint processing
        if len(rendered_images) > 0:
            image_shape = rendered_images[0].shape[:2]  # (height, width)
        else:
            raise ValueError("No rendered images provided")

        # Rotate and center-crop image
        rendered_images_transformed = []
        for img in rendered_images:
            # note: rotate_image calls scipy.ndimage.rotate which expects angle in
            # degrees and rotates counter-clockwise, hence the negative sign
            rotated_img = rotate_image(img, -rotation_angle)
            img_transformed, start_col, start_row = center_square_crop_image(
                rotated_img, crop_size
            )
            rendered_images_transformed.append(img_transformed)

        # Add all derived variables to a single dict
        derived_variables = {}

        # Gather keypoint positions in coordinates and rotate/center-crop accordingly
        keypoints_pos_dict_world_raw, keypoints_pos_dict_camera_raw = (
            extract_body_segment_positions(
                h5_file, frame_idx, "pos_atparent", constants.keypoint_segments_nmf
            )
        )
        keypoints_pos_dict_world_rotated = rotate_keypoint_positions_world(
            keypoints_pos_dict_world_raw, rotation_matrix
        )
        keypoints_pos_lookup_cam_rotated = rotate_keypoint_positions_camera(
            keypoints_pos_dict_camera_raw,
            rotation_matrix,
            image_shape=image_shape,
            start_col=start_col,
            start_row=start_row,
        )
        for body_segment_name, rotated_pos in keypoints_pos_dict_world_rotated.items():
            derived_variables[f"keypoint_pos_world_{body_segment_name}"] = rotated_pos
        for body_segment_name, rotated_pos in keypoints_pos_lookup_cam_rotated.items():
            derived_variables[f"keypoint_pos_camera_{body_segment_name}"] = rotated_pos

        # Save object segmentation masks
        seg_labels = segment_label_parser(rendered_images_transformed)

        # Transform mesh states to camera's coordinate system
        pos_glob = h5_file["body_segment_states/pos_global"][frame_idx, :, :]
        quat_glob = h5_file["body_segment_states/quat_global"][frame_idx, :, :]
        cam_projmat = h5_file["camera_matrix"][frame_idx, :, :]  # 3x4 mat from mujoco
        derived_variables["pos_rel_cam"], derived_variables["quat_rel_cam"] = (
            calculate_6dpose_relative_to_camera(pos_glob, quat_glob, cam_projmat)
        )

        return rendered_images_transformed, derived_variables, seg_labels


def calculate_6dpose_relative_to_camera(pos_global, quat_global, cam_projmat):
    """Convert global mesh positions and orientations to camera-relative coordinates.

    Args:
        pos_global: (num_segments, 3) array of global positions
        quat_global: (num_segments, 4) array of global quaternions (scalar first)
        cam_projmat: (3, 4) camera projection matrix

    Returns:
        pos_rel_cam: (num_segments, 3) array of positions relative to camera
        quat_rel_cam: (num_segments, 4) array of quaternions relative to camera
    """
    assert pos_global.shape[0] == quat_global.shape[0], "Number of segments mismatch."
    n_segments = pos_global.shape[0]

    # Decompose camera projection matrix to get camera intrinsics, rotation, translation
    cam_intrinsics, cam_rotation = rq(cam_projmat[:, :3])
    _sign_multiplier = np.diag(np.sign(np.diag(cam_intrinsics)))
    cam_intrinsics = cam_intrinsics @ _sign_multiplier
    cam_rotation = _sign_multiplier @ cam_rotation
    if np.linalg.det(cam_rotation) < 0:
        cam_rotation = -cam_rotation  # ensure proper rotation matrix (det = 1)
        cam_intrinsics = -cam_intrinsics
    cam_translation = np.linalg.inv(cam_intrinsics) @ cam_projmat[:, 3]

    # Compute rotation from world to camera coordinates
    rot_world_to_cam = Rotation.from_matrix(cam_rotation)

    # Convert each segment's position and orientation
    pos_rel_cam = np.zeros_like(pos_global)
    quat_rel_cam = np.zeros_like(quat_global)
    for seg_idx in range(n_segments):
        this_pos_glob = pos_global[seg_idx, :]
        this_quat_glob = quat_global[seg_idx, :]

        this_pos_rel_cam = cam_rotation @ this_pos_glob + cam_translation
        mesh_rot = Rotation.from_quat(this_quat_glob, scalar_first=True)
        mesh_rot_rel_cam = rot_world_to_cam * mesh_rot
        this_quat_rel_cam = mesh_rot_rel_cam.as_quat(scalar_first=True)

        pos_rel_cam[seg_idx, :] = this_pos_rel_cam
        quat_rel_cam[seg_idx, :] = this_quat_rel_cam

    return pos_rel_cam, quat_rel_cam


def process_subsegment(
    frames_by_color_coding: list[list[np.ndarray]],
    segment_h5_file_path: Path,
    frames_range: tuple[int, int],
    processed_subsegment_dir: Path,
    fps: int | float,
    crop_size: int = 464,
    num_color_codings: int = 2,
    n_jobs: int = -1,
) -> None:
    frame_idx_start, frame_idx_end = frames_range
    num_frames = frame_idx_end - frame_idx_start

    # Filter image data
    subsegment_frames_by_color_coding = [
        frames[frame_idx_start:frame_idx_end] for frames in frames_by_color_coding
    ]

    processed_subsegment_dir.mkdir(parents=True, exist_ok=True)

    # Create segment label parser
    segment_label_parser = SegmentLabelParser()

    # Process frames in parallel
    # Prepare arguments for each frame
    frame_args = []
    for i_frame_within_subsegment in range(num_frames):
        renderings_per_frame = [
            subsegment_frames_by_color_coding[i_color_code][i_frame_within_subsegment]
            for i_color_code in range(num_color_codings)
        ]
        frame_args.append(
            (
                renderings_per_frame,
                segment_h5_file_path,
                frame_idx_start + i_frame_within_subsegment,
                crop_size,
                segment_label_parser,
            )
        )
    # Parallel execution with joblib
    # Use 'loky' backend for CPU-intensive image processing operations
    parallel_executor = Parallel(n_jobs=n_jobs, backend="loky")
    effective_n_jobs = parallel_executor._effective_n_jobs()
    print(
        f"Processing {num_frames} frames in parallel using {n_jobs} cores "
        f"(effectively {effective_n_jobs} cores)..."
    )
    results = parallel_executor(
        delayed(process_single_frame)(*args)
        for args in tqdm(frame_args, desc="Processing frames", disable=None)
    )

    # Extract results
    transformed_images_all = []
    seg_labels_all = []
    derived_variables_by_key = defaultdict(list)
    for transformed_images, derived_variables, seg_labels in results:
        transformed_images_all.append(transformed_images)
        for key, value in derived_variables.items():
            derived_variables_by_key[key].append(value)
        seg_labels_all.append(seg_labels)

    # Write processed images to disk as a video
    for i_color_code in range(num_color_codings):
        output_video_path = (
            processed_subsegment_dir
            / f"processed_nmf_sim_render_colorcode_{i_color_code}.mp4"
        )
        with imageio.get_writer(
            str(output_video_path),
            fps=fps,
            codec="libx264",
            quality=10,  # 10 is highest for imageio, lower is lower quality
            ffmpeg_params=[
                "-crf",
                "18",
                "-preset",
                "slow",
            ],  # lower crf = higher quality
        ) as writer:
            for transformed_images in tqdm(
                transformed_images_all, desc="Writing frames", disable=None
            ):
                writer.append_data(transformed_images[i_color_code])

    out_h5_path = processed_subsegment_dir / "processed_simulation_data.h5"
    with h5py.File(segment_h5_file_path, "r") as source_h5_file:
        with h5py.File(out_h5_path, "w") as subsegment_h5_file:
            # Copy all original data
            raw_group = subsegment_h5_file.create_group("raw")
            for key in source_h5_file.keys():
                source_h5_file.copy(key, raw_group)
            raw_group.attrs["n_timesteps"] = source_h5_file.attrs["n_timesteps"]
            raw_group.attrs["description"] = (
                "Data collected during the full NeuroMechFly simulation that this "
                "subsegment is part of."
            )

            # Create group for postprocessed data
            postprocessed_group = subsegment_h5_file.create_group("postprocessed")
            postprocessed_group.attrs["n_timesteps"] = num_frames
            postprocessed_group.attrs["description"] = (
                "Variables derived from the raw simulation data, including DoF angles, "
                "3D keypoint positions, body segment positions and orientations, and "
                "2D segmentation labels from the camera's perspective. The images have "
                "been transformed to be centered on the fly, cropped to a square, and "
                "rotated so that the fly faces up. The variables in this data group "
                "correspond to the processed images after these transformations."
            )
            postprocessed_group.attrs["frame_indices_in_full_simulation"] = [
                frame_idx_start,
                frame_idx_end,
            ]

            # Add selected subsets of DoF angles
            dof_pos_ds = postprocessed_group.create_dataset(
                "dof_angles",
                data=source_h5_file["dof_angles"][frame_idx_start:frame_idx_end, :],
                dtype="float32",
            )
            dof_pos_ds.attrs["description"] = (
                "Joint angles for all DoFs tracked in the simulation. This dataset has "
                "shape (n_timesteps, n_dofs). The order of the DoFs is given in the "
                "'keys' attribute."
            )
            dof_pos_ds.attrs["keys"] = source_h5_file["dof_angles"].attrs["keys"]
            dof_pos_ds.attrs["units"] = "radians"

            # Add derived variables
            keypoint_pos_group = postprocessed_group.create_group("keypoint_pos")
            for ref_frame in ["camera", "world"]:
                data_block = np.empty(
                    (num_frames, len(constants.keypoint_segments_nmf), 3),
                    dtype="float32",
                )
                for seg_id, body_segment in enumerate(constants.keypoint_segments_nmf):
                    key = f"keypoint_pos_{ref_frame}_{body_segment}"
                    values = np.array(derived_variables_by_key[key])
                    data_block[:, seg_id, :] = values

                pos_ds = keypoint_pos_group.create_dataset(
                    f"{ref_frame}_coords", data=data_block, dtype="float32"
                )
                pos_ds.attrs["keys"] = constants.keypoint_segments_nmf
                pos_ds.attrs["description"] = (
                    f"Keypoint positions in {ref_frame} coordinates. Shape is "
                    "(num_frames, num_keypoints, 3). See the `.attrs['keys']` for the "
                    "order of keypoints."
                )
            keypoint_pos_group.attrs["keys"] = constants.keypoint_segments_nmf
            keypoint_pos_group.attrs["description"] = (
                "This group contains positions of joint keypoints in the rotated image "
                "centered around the fly, cropped, and rotated so that the fly faces "
                'upwards. Keypoint order is given in <this_group>.attrs["keys"].'
            )

            # Add segmentation labels
            seg_labels_all = np.array(seg_labels_all)
            assert (
                seg_labels_all.max() < 256
            ), "Unique segmentation labels exceed 255, cannot be stored as uint8."
            assert (
                seg_labels_all.min() >= 0
            ), "Segmentation labels contain negative values, cannot be stored as uint8."
            seg_labels_ds = postprocessed_group.create_dataset(
                "segmentation_labels",
                data=seg_labels_all,
                dtype="uint8",
                compression="lzf",  # lzf is faster than gzip - good for frequent access
                shuffle=True,
            )
            seg_labels_ds.attrs["keys"] = segment_label_parser.label_keys
            seg_labels_ds.attrs["description"] = (
                "Segmentation labels (i.e. body segment IDs) for each pixel in the "
                "processed (i.e. centered, cropped, rotated) images. Shape is "
                "(num_frames, height, width). See `.attrs['keys']` on this dataset "
                "for the mapping from label IDs (pixel values) to body segment names."
            )

            # Add mesh state labels
            seg_states_grp = postprocessed_group.create_group("mesh_pose6d_rel_camera")
            seg_states_grp.create_dataset(
                "pos_rel_cam",
                data=np.array(derived_variables_by_key["pos_rel_cam"]),
                dtype="float32",
                compression="lzf",
            )
            seg_states_grp.create_dataset(
                "quat_rel_cam",
                data=np.array(derived_variables_by_key["quat_rel_cam"]),
                dtype="float32",
                compression="lzf",
            )
            seg_states_grp.attrs.update(source_h5_file["body_segment_states"].attrs)


def _draw_pose_2d_and_3d(
    ax_pose2d: plt.Axes,
    ax_pose3d: plt.Axes,
    *,
    frame_index: int,
    frame: np.ndarray,
    processed_sim_data_path: Path,
    camera_elevation: float,
    azimuth_rotation_period: float,
    max_abs_azimuth: float,
):
    # Calculate azimuth for this frame
    azimuth = (
        np.cos(2 * np.pi * frame_index / azimuth_rotation_period) * max_abs_azimuth
    )
    ax_pose3d.view_init(elev=camera_elevation, azim=azimuth)

    with h5py.File(processed_sim_data_path, "r") as h5_file:
        # Plot 2D image
        ax_pose2d.imshow(frame)

        # Overlay keypoints on 2D image (note that coords are in row, col, depth)
        # Legs
        keypoint_pos_cam_ds = h5_file["postprocessed/keypoint_pos/camera_coords"]
        keypoints = keypoint_pos_cam_ds.attrs["keys"].tolist()
        for leg in constants.legs:
            color = constants.kchain_plotting_colors[leg]
            all_positions = []
            for kpt in constants.leg_keypoints_canonical:
                segment_name = constants.keypoint_name_lookup_canonical_to_nmf[kpt]
                keypoint_idx = keypoints.index(f"{leg}{segment_name}")
                pos = keypoint_pos_cam_ds[frame_index, keypoint_idx, :]
                all_positions.append(pos)
            all_positions = np.array(all_positions)
            ax_pose2d.plot(
                all_positions[:, 0], all_positions[:, 1], color=color, linewidth=2
            )
        # Antenna
        for side in "LR":
            segment_name = f"{side}Pedicel"
            keypoint_idx = keypoints.index(segment_name)
            pos = keypoint_pos_cam_ds[frame_index, keypoint_idx, :]
            color = constants.kchain_plotting_colors[f"{side}Antenna"]
            ax_pose2d.plot(pos[0], pos[1], marker="o", color=color, markersize=5)

        # Plot 3D keypoints
        keypoint_pos_world_ds = h5_file["postprocessed/keypoint_pos/world_coords"]
        # Legs
        for leg in constants.legs:
            color = constants.kchain_plotting_colors[leg]
            all_positions = []
            for kpt in constants.leg_keypoints_canonical:
                segment_name = constants.keypoint_name_lookup_canonical_to_nmf[kpt]
                keypoint_idx = keypoints.index(f"{leg}{segment_name}")
                pos = keypoint_pos_world_ds[frame_index, keypoint_idx, :]
                all_positions.append(pos)
            all_positions = np.array(all_positions)
            ax_pose3d.plot(
                all_positions[:, 0],
                all_positions[:, 1],
                all_positions[:, 2],
                marker="o",
                color=color,
                linewidth=2,
            )
        # Antenna
        for side in "LR":
            segment_name = f"{side}Pedicel"
            keypoint_idx = keypoints.index(segment_name)
            pos = keypoint_pos_world_ds[frame_index, keypoint_idx, :]
            color = constants.kchain_plotting_colors[f"{side}Antenna"]
            ax_pose3d.plot(
                pos[0], pos[1], pos[2], marker="o", color=color, markersize=5
            )

        ax_pose2d.set_axis_off()
        ax_pose3d.set_xlabel("anterior-posterior")
        ax_pose3d.set_ylabel("lateral")
        ax_pose3d.set_zlabel("dorsal-ventral")
        ax_pose3d.set_xlim(-2, 3)
        ax_pose3d.set_ylim(-2, 2)
        ax_pose3d.set_zlim(-0.5, 2)
        ax_pose3d.set_aspect("equal")


def _draw_segmentation_labels_map(
    ax_seglabel: plt.Axes,
    seg_labels: np.ndarray,
    seg_labels_color_palette: list[np.ndarray],
):
    assert seg_labels.max() <= len(seg_labels_color_palette) - 1, (
        f"seg_labels has max value {seg_labels.max()} but only "
        f"{len(seg_labels_color_palette)} colors are available."
    )
    visualized_map = np.zeros((*seg_labels.shape, 3), dtype=np.uint8)
    for i in range(0, len(seg_labels_color_palette)):
        visualized_map[seg_labels == i] = np.array(seg_labels_color_palette[i]) * 255

    ax_seglabel.imshow(visualized_map)
    ax_seglabel.axis("off")


def visualize_single_frame(
    frame_index: int,
    frame: np.ndarray,
    processed_sim_data_path: Path,
    seg_labels: np.ndarray,
    temp_dir: Path,
    camera_elevation: float,
    azimuth_rotation_period: float,
    max_abs_azimuth: float,
    seg_labels_color_palette: list[np.ndarray],
):
    """
    Worker function for parallel frame visualization.

    Args:
        frame_index: Index of the frame for proper ordering
        frame: Input image frame
        processed_sim_data_path: Path to the processed simulation data
            file (HDF5)
        seg_labels: Segmentation labels for the frame
        temp_dir: Temporary directory for saving frames
        camera_elevation: 3D plot camera elevation
        azimuth_rotation_period: Period for azimuth rotation
        max_abs_azimuth: Maximum azimuth angle

    Returns:
        Tuple of (frame_index, path_to_saved_frame)
    """
    # Create a new figure for this worker (thread-safe)
    fig = plt.figure(figsize=(18, 6))
    ax_pose2d = fig.add_subplot(1, 3, 1)
    ax_pose3d = fig.add_subplot(1, 3, 2, projection="3d")
    ax_seglabel = fig.add_subplot(1, 3, 3)
    ax_pose2d.set_title("2D pose overlay")
    ax_pose3d.set_title("3D pose")
    ax_seglabel.set_title("Segmentation labels")

    # Plot 2D and 3D pose
    _draw_pose_2d_and_3d(
        ax_pose2d,
        ax_pose3d,
        frame_index=frame_index,
        frame=frame,
        processed_sim_data_path=processed_sim_data_path,
        camera_elevation=camera_elevation,
        azimuth_rotation_period=azimuth_rotation_period,
        max_abs_azimuth=max_abs_azimuth,
    )

    # Draw segmentation labels map
    _draw_segmentation_labels_map(ax_seglabel, seg_labels, seg_labels_color_palette)

    # Save the figure
    viz_frame_path = temp_dir / f"frame_{frame_index:06d}.png"
    fig.savefig(viz_frame_path)
    plt.close(fig)

    return frame_index, viz_frame_path


def visualize_subsegment(
    processed_subsegment_dir: Path,
    render_fps: float | None = None,
    camera_elevation: float = 30.0,
    max_abs_azimuth: float = 30.0,
    azimuth_rotation_period: float = 300.0,
    n_jobs: int = -1,  # Default to all available cores
) -> None:
    # Load video frames
    frames, input_fps = load_video_frames(
        processed_subsegment_dir / "processed_nmf_sim_render_colorcode_0.mp4"
    )
    if render_fps is None:
        render_fps = input_fps

    # Find processed simulation data
    processed_data_path = processed_subsegment_dir / "processed_simulation_data.h5"
    with h5py.File(processed_data_path, "r") as h5_file:
        ds = h5_file["postprocessed/segmentation_labels"]
        seg_labels_all = ds[...]
        keys = ds.attrs["keys"].tolist()

    # Define color palette for segmentation labels visualization
    max_num_labels = len(keys)
    assert keys[0] == "Background"
    assert keys[1] == "OtherSegments"
    assert keys[2] == "Thorax"
    color_palette = get_segmentation_color_palette(max_num_labels)

    # Make temp directory for visualizations
    temp_dir = Path(
        tempfile.mkdtemp(prefix="visualizations_", dir=os.environ.get("TMPDIR", "/tmp"))
    )
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Visualize frames in parallel
    parallel_executor = Parallel(n_jobs=n_jobs, backend="loky")
    effective_n_jobs = parallel_executor._effective_n_jobs()
    print(
        f"Visualizing {len(frames)} frames in parallel using {n_jobs} cores "
        f"(effectively {effective_n_jobs} cores)..."
    )
    results = parallel_executor(
        delayed(visualize_single_frame)(
            i,
            frame,
            processed_data_path,
            seg_labels_all[i],
            temp_dir,
            camera_elevation,
            azimuth_rotation_period,
            max_abs_azimuth,
            seg_labels_color_palette=color_palette,
        )
        for i, frame in tqdm(
            enumerate(frames),
            total=len(frames),
            desc="Visualizing frames",
            disable=None,
        )
    )

    # Sort results by frame index to maintain order
    results.sort(key=lambda x: x[0])
    viz_frames_paths = [path for _, path in results]

    # Merge video
    with imageio.get_writer(
        str(processed_subsegment_dir / "visualization.mp4"),
        fps=render_fps,
        codec="libx264",
        quality=10,  # 10 is highest for imageio, lower is lower quality
        ffmpeg_params=["-crf", "18", "-preset", "slow"],  # lower crf = higher quality
    ) as writer:
        for img_path in tqdm(viz_frames_paths, desc="Writing frames", disable=None):
            img = imageio.imread(img_path)
            writer.append_data(img)

    # Cleanup
    shutil.rmtree(temp_dir)


def select_subsegments(
    upward_cardinal_vectors: np.ndarray,
    max_tilt_angle_deg: float,
    mask_morph_closing_size_sec: float,
    min_subsegment_duration_sec: float,
    timestep: float,
):
    z_component = upward_cardinal_vectors[:, 2]
    angles = np.arccos(z_component)
    upright_mask = angles <= np.deg2rad(max_tilt_angle_deg)
    kernel_size_frames = int(0.5 * mask_morph_closing_size_sec / timestep) * 2 + 1
    morph_closing_kernel = np.ones(kernel_size_frames, dtype=bool)  # size must be odd
    upright_mask = ndimage.binary_closing(upright_mask, structure=morph_closing_kernel)

    # Find continuous periods in which the fly is upright
    min_subsegment_duration_frames = int(min_subsegment_duration_sec / timestep)
    labeled, n_features = ndimage.label(upright_mask)
    subsegments_boundaries = []
    # Note: ndimage.label output is 1-indexed (0 is just background)
    for subsegment_id in range(1, n_features + 1):
        indices = np.where(labeled == subsegment_id)[0]
        if len(indices) > min_subsegment_duration_frames:
            subsegments_boundaries.append((indices[0], indices[-1] + 1))

    return subsegments_boundaries


def postprocess_segment(
    recording_dir: Path,
    start_frame: int = 0,
    end_frame: int = -1,
    max_tilt_angle_deg: float = 30.0,
    mask_morph_closing_size_sec: float = 0.03,
    min_subsegment_duration_sec: float = 0.1,
    image_crop_size: int = 464,
    visualize: bool = False,
    camera_elevation: float = 30.0,
    max_abs_azimuth: float = 30.0,
    azimuth_rotation_period: float = 300.0,
    n_jobs: int = -1,
    num_color_codings: int = 2,
):
    if not recording_dir.is_dir():
        raise FileNotFoundError(f"{recording_dir} is not a directory.")

    segment_h5_file_path = recording_dir / "simulation_data.h5"

    # Load video frames
    frames_by_color_coding, fps, num_frames = read_videos(
        recording_dir, num_color_codings
    )
    for i, frames in enumerate(frames_by_color_coding):
        frames_by_color_coding[i] = frames[start_frame:end_frame]

    with h5py.File(segment_h5_file_path, "r") as h5_file:
        upward_cardinal_vectors = h5_file["cardinal_vectors/up"][
            start_frame:end_frame, :
        ]
        timestep = h5_file["sim_time"][1] - h5_file["sim_time"][0]

        subsegments_boundaries = select_subsegments(
            upward_cardinal_vectors,
            max_tilt_angle_deg,
            mask_morph_closing_size_sec,
            min_subsegment_duration_sec,
            timestep=timestep,
        )
        print(
            f"Found {len(subsegments_boundaries)} subsegments "
            "in which the fly is upright."
        )

        # Process each subsegment
        for i, (start, end) in enumerate(subsegments_boundaries):
            print(
                f"Processing subsegment {i + 1}/{len(subsegments_boundaries)} "
                f"(frames {start}:{end})"
            )

            # Process the subsegment
            output_dir = recording_dir / f"subsegment_{i:03d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            process_subsegment(
                frames_by_color_coding,
                segment_h5_file_path,
                (start, end),
                output_dir,
                fps,
                image_crop_size,
                n_jobs=n_jobs,
            )

            # Visualize the subsegment if requested
            if visualize:
                visualize_subsegment(
                    processed_subsegment_dir=output_dir,
                    render_fps=fps,
                    camera_elevation=camera_elevation,
                    max_abs_azimuth=max_abs_azimuth,
                    azimuth_rotation_period=azimuth_rotation_period,
                    n_jobs=n_jobs,
                )


def read_videos(recording_dir, num_color_codings):
    frames_by_color_coding = []
    fps = None
    num_frames = None

    for color_code_idx in range(num_color_codings):
        video_path = recording_dir / f"nmf_sim_render_colorcode_{color_code_idx}.mp4"
        my_frames, my_fps = load_video_frames(video_path)
        frames_by_color_coding.append(my_frames)

        print(f"Loaded {len(my_frames)} frames from {video_path} at {my_fps} FPS.")

        # Check for consistency
        if fps is None:
            fps = my_fps
        else:
            assert (
                fps == my_fps
            ), "FPS mismatch between videos of different color codings"
        if num_frames is None:
            num_frames = len(my_frames)
        else:
            assert num_frames == len(
                my_frames
            ), "number of frames mismatch between videos of different color codings"

    return frames_by_color_coding, fps, num_frames
