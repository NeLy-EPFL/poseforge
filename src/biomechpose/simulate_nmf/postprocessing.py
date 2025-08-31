import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import imageio.v2 as imageio
import shutil
import tempfile
import os
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

import biomechpose.simulate_nmf.simulate as simulate
from biomechpose.simulate_nmf.utils import (
    keypoint_name_lookup_canonical_to_nmf,
    kchain_plotting_colors,
)

keypoint_segments = [
    f"{side}{pos}{keypoint_name_lookup_canonical_to_nmf[link]}"
    for side in "LR"
    for pos in "FMH"
    for link in ["ThC", "CTr", "FTi", "TiTa", "Claw"]
] + ["LPedicel", "RPedicel"]
legs = [f"{side}{pos}" for side in "LR" for pos in "FMH"]
leg_keypoints = ["ThC", "CTr", "FTi", "TiTa", "Claw"]


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
                    color0 = nmf_rendered_colors[simulate.color_by_link[link]]
                    color1 = nmf_rendered_colors[simulate.color_by_kinematic_chain[leg]]
                    color_6d = np.array(list(color0) + list(color1))
                    label = f"{leg}{link}"
                    self.label_keys.append(label)
                    self.label_colors_6d.append(color_6d)

        # Antennas
        for side in "LR":
            color0 = nmf_rendered_colors[simulate.color_by_link["Antenna"]]
            color1 = nmf_rendered_colors[simulate.color_by_kinematic_chain[side]]
            color_6d = np.array(list(color0) + list(color1))
            label = f"{side}Antenna"
            self.label_keys.append(label)
            self.label_colors_6d.append(color_6d)

        self.label_colors_6d = np.array(self.label_colors_6d)

    def __call__(self, images_by_color_coding: list[np.ndarray]):
        assert (
            len(images_by_color_coding) == 2
        ), "Expecting two images each with a different color coding."
        assert (
            images_by_color_coding[0].shape == images_by_color_coding[1].shape
        ), "Color coding images must have the same shape."
        if not isinstance(images_by_color_coding, list):
            images_by_color_coding = [
                images_by_color_coding[0, :, :, :],
                images_by_color_coding[1, :, :, :],
            ]

        image_6d = np.concatenate(images_by_color_coding, axis=-1)
        sq_distances = np.sum(
            (image_6d[:, :, None, :] - self.label_colors_6d[None, None, :, :]) ** 2,
            axis=-1,
        )
        label_indices = np.argmin(sq_distances, axis=-1)

        return label_indices


def load_video_frames(video_path: Path) -> list[np.ndarray]:
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


def get_rotation_angle_and_matrix(forward_vector: np.ndarray) -> np.ndarray:
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


def center_square_crop_image(image: np.ndarray, side_length) -> np.ndarray:
    """Crop the image to a square of given side length, centered on the image."""
    height, width = image.shape[:2]
    start_col = (width - side_length) // 2
    start_row = (height - side_length) // 2
    cropped_image = image[
        start_row : start_row + side_length, start_col : start_col + side_length
    ]
    return cropped_image, start_col, start_row


def extract_body_segment_positions(
    df_row: pd.Series, sensor_type: str, body_segments: list[str] | None
):
    # Compile list of body segments
    if body_segments is None:
        prefix = f"body_seg_{sensor_type}_"
        body_segments = [
            col.replace(prefix, "") for col in df_row.index if col.startswith(prefix)
        ]

    # Get positions of each keypoint
    keys = [f"body_seg_{sensor_type}_{seg}" for seg in body_segments]
    pos_world = np.array(df_row[keys].to_list()).T
    pos_world_homogeneous = np.vstack(
        [pos_world, np.ones((1, pos_world.shape[1]))]
    )  # (4, num_keypoints)

    # Project to camera coordinates using the working approach from the snippet
    camera_matrix = df_row["camera_matrix"]  # This is 3x4 from MuJoCo

    # Apply camera matrix directly (this gives us the projected coordinates)
    pos_camera = camera_matrix @ pos_world_homogeneous  # (3, num_keypoints)

    # The third component is depth, first two are already in image coordinates

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
):
    """
    Apply rotation and cropping transformations to camera/image coordinates.
    Uses a simplified approach based on the working code snippet.

    Args:
        keypoints_pos_lookup: Dict mapping segment names to (x, y, depth) positions
        rotation_matrix: 3x3 rotation matrix used for coordinate transformation
        image_shape: (height, width) of the original image before rotation
        start_col, start_row: Cropping offsets after rotation
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
):
    keypoints_pos_lookup_rotated = {}
    for body_segment_name, pos_before_rotation in keypoints_pos_lookup.items():
        # Apply the rotation matrix to the keypoints
        pos_after_rotation = rotation_matrix @ pos_before_rotation
        keypoints_pos_lookup_rotated[body_segment_name] = pos_after_rotation

    return keypoints_pos_lookup_rotated


def process_single_frame(
    rendered_images: np.ndarray,
    df_row: pd.Series,
    crop_size: int,
    segment_label_parser: SegmentLabelParser,
):
    # Get rotation angle and rotation matrix
    forward_vector = df_row["cardinal_vector_forward"]
    rotation_angle, rotation_matrix = get_rotation_angle_and_matrix(forward_vector)

    # Get image shape before rotation for keypoint processing
    if len(rendered_images) > 0:
        image_shape = rendered_images[0].shape[:2]  # (height, width)
    else:
        raise ValueError("No rendered images provided")

    # Rotate and center-crop image
    rendered_images_transformed = []
    for img in rendered_images:
        # note: rotate_image calls scipy.ndimage.rotate which expects angle in degrees
        # and rotates counter-clockwise, hence the negative sign
        rotated_img = rotate_image(img, -rotation_angle)
        img_transformed, start_col, start_row = center_square_crop_image(
            rotated_img, crop_size
        )
        rendered_images_transformed.append(img_transformed)

    # Add all derived variables to a single dict
    all_derived_variables = {}

    # Gather keypoint positions in coordinates and rotate/center-crop accordingly
    keypoints_pos_dict_world_raw, keypoints_pos_dict_camera_raw = (
        extract_body_segment_positions(df_row, "pos_atparent", keypoint_segments)
    )
    keypoints_pos_dict_world_rotated = rotate_keypoint_positions_world(
        keypoints_pos_dict_world_raw, rotation_matrix
    )
    keypoints_pos_lookup_camera_rotated = rotate_keypoint_positions_camera(
        keypoints_pos_dict_camera_raw,
        rotation_matrix,
        image_shape=image_shape,
        start_col=start_col,
        start_row=start_row,
    )
    for body_segment_name, rotated_pos in keypoints_pos_dict_world_rotated.items():
        all_derived_variables[f"keypoint_pos_world_{body_segment_name}"] = rotated_pos
    for body_segment_name, rotated_pos in keypoints_pos_lookup_camera_rotated.items():
        all_derived_variables[f"keypoint_pos_camera_{body_segment_name}"] = rotated_pos

    # Save object segmentation masks
    seg_labels = segment_label_parser(rendered_images_transformed)

    return rendered_images_transformed, all_derived_variables, seg_labels


def process_subsegment(
    subsegment_frames_by_color_coding: list[list[np.ndarray]],
    kinematic_states_df: pd.DataFrame,
    processed_subsegment_dir: Path,
    fps: int,
    crop_size: int = 464,
    num_color_codings: int = 2,
    n_jobs: int = -1,
) -> None:
    kinematic_states_df = kinematic_states_df.copy()
    num_frames = len(subsegment_frames_by_color_coding[0])
    if len(kinematic_states_df) != num_frames:
        raise ValueError(
            f"Number of frames in video ({num_frames}) does not match "
            f"number of kinematic states ({len(kinematic_states_df)})."
        )

    processed_subsegment_dir.mkdir(parents=True, exist_ok=True)

    # Create segment label parser
    segment_label_parser = SegmentLabelParser()

    # Process frames in parallel
    print(f"Processing {num_frames} frames in parallel using {n_jobs} cores...")
    # Prepare arguments for each frame
    frame_args = []
    for i_frame in range(num_frames):
        renderings_per_frame = [
            subsegment_frames_by_color_coding[i_color_code][i_frame]
            for i_color_code in range(num_color_codings)
        ]
        df_row = kinematic_states_df.iloc[i_frame]
        frame_args.append(
            (renderings_per_frame, df_row, crop_size, segment_label_parser)
        )

    # Process frames in parallel
    # Use 'loky' backend for CPU-intensive image processing operations
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_single_frame)(*args)
        for args in tqdm(frame_args, desc="Processing frames", disable=None)
    )

    # Extract results
    transformed_images_all = []
    seg_labels_all = []
    derived_variables_by_column = defaultdict(list)
    for transformed_images, derived_variables, seg_labels in results:
        transformed_images_all.append(transformed_images)
        for key, value in derived_variables.items():
            derived_variables_by_column[key].append(value)
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

    # Update dataframe of kinematic states with transformed keypoint positions
    # Use pd.concat to avoid DataFrame fragmentation warnings
    new_columns_df = pd.DataFrame(
        derived_variables_by_column, index=kinematic_states_df.index
    )
    kinematic_states_df = pd.concat([kinematic_states_df, new_columns_df], axis=1)

    kinematic_states_df.to_pickle(
        processed_subsegment_dir / "processed_kinematic_states.pkl"
    )

    # Save segmentation labels
    np.savez_compressed(
        processed_subsegment_dir / "segmentation_labels.npz",
        labels=seg_labels_all,
        label_keys=segment_label_parser.label_keys,
    )


def visualize_single_frame(
    frame_index: int,
    frame: np.ndarray,
    df_row: pd.Series,
    temp_dir: Path,
    camera_elevation: float,
    azimuth_rotation_period: float,
    max_abs_azimuth: float,
):
    """
    Worker function for parallel frame visualization.

    Args:
        frame_index: Index of the frame for proper ordering
        frame: Input image frame
        kinematic_entry: DataFrame row with kinematic state data
        temp_dir: Temporary directory for saving frames
        camera_elevation: 3D plot camera elevation
        azimuth_rotation_period: Period for azimuth rotation
        max_abs_azimuth: Maximum azimuth angle

    Returns:
        Tuple of (frame_index, path_to_saved_frame)
    """
    # Create a new figure for this worker (thread-safe)
    fig = plt.figure(figsize=(12, 6))
    ax_2d = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

    try:
        # Calculate azimuth for this frame
        azimuth = (
            np.cos(2 * np.pi * frame_index / azimuth_rotation_period) * max_abs_azimuth
        )
        ax_3d.view_init(elev=camera_elevation, azim=azimuth)

        # Plot 2D image
        ax_2d.imshow(frame)

        # Overlay keypoints on 2D image (note that coords are in row, col, depth)
        # Legs
        for leg in legs:
            color = kchain_plotting_colors[leg]
            all_positions = []
            for kpt in leg_keypoints:
                segment_name = keypoint_name_lookup_canonical_to_nmf[kpt]
                all_positions.append(df_row[f"keypoint_pos_camera_{leg}{segment_name}"])
            all_positions = np.array(all_positions)
            ax_2d.plot(
                all_positions[:, 0], all_positions[:, 1], color=color, linewidth=2
            )
        # Antenna
        for side in "LR":
            color = kchain_plotting_colors[f"{side}Antenna"]
            pos = df_row[f"keypoint_pos_camera_{side}Pedicel"]
            ax_2d.plot(pos[0], pos[1], marker="o", color=color, markersize=5)

        # Plot 3D keypoints
        for leg in legs:
            color = kchain_plotting_colors[leg]
            all_positions = []
            for kpt in leg_keypoints:
                segment_name = keypoint_name_lookup_canonical_to_nmf[kpt]
                all_positions.append(df_row[f"keypoint_pos_world_{leg}{segment_name}"])
            all_positions = np.array(all_positions)
            ax_3d.plot(
                all_positions[:, 0],
                all_positions[:, 1],
                all_positions[:, 2],
                marker="o",
                color=color,
                linewidth=2,
            )
        # Antenna
        for side in "LR":
            color = kchain_plotting_colors[f"{side}Antenna"]
            pos = df_row[f"keypoint_pos_world_{side}Pedicel"]
            ax_3d.plot(pos[0], pos[1], pos[2], marker="o", color=color, markersize=5)

        ax_2d.set_axis_off()
        ax_3d.set_xlabel("anterior-posterior")
        ax_3d.set_ylabel("lateral")
        ax_3d.set_zlabel("dorsal-ventral")
        ax_3d.set_xlim(-2, 3)
        ax_3d.set_ylim(-2, 2)
        ax_3d.set_zlim(-0.5, 2)
        ax_3d.set_aspect("equal")

        # Save the figure
        viz_frame_path = temp_dir / f"frame_{frame_index:06d}.png"
        fig.savefig(viz_frame_path)

        return frame_index, viz_frame_path

    finally:
        # Always close the figure to free memory
        plt.close(fig)


def visualize_subsegment(
    processed_subsegment_dir: Path,
    fps: int,
    camera_elevation: float = 30.0,
    max_abs_azimuth: float = 30.0,
    azimuth_rotation_period: float = 300.0,
    n_jobs: int = -1,  # Default to all available cores
) -> None:
    # Load video frames
    frames, fps = load_video_frames(
        processed_subsegment_dir / "processed_nmf_sim_render_colorcode_0.mp4"
    )

    # Load kinematic states history
    kinematic_states_df = pd.read_pickle(
        processed_subsegment_dir / "processed_kinematic_states.pkl"
    )

    if len(kinematic_states_df) != len(frames):
        raise ValueError(
            f"Number of frames in video ({len(frames)}) does not match "
            f"number of kinematic states ({len(kinematic_states_df)})."
        )

    # Make temp directory for visualizations
    temp_dir = Path(
        tempfile.mkdtemp(prefix="visualizations_", dir=os.environ.get("TMPDIR", "/tmp"))
    )
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Visualize frames in parallel
    print(
        f"Visualizing {len(frames)} frames in parallel using "
        f"{n_jobs if n_jobs > 0 else 'all available'} cores..."
    )
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(visualize_single_frame)(
            i,
            frame,
            kinematic_states_df.iloc[i],
            temp_dir,
            camera_elevation,
            azimuth_rotation_period,
            max_abs_azimuth,
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
        fps=fps,
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
    upward_cardinal_vectors: np.array,
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
    for subsegment_id in range(1, n_features + 1):  # ndimage.label output is 1-indexed
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

    # Load video frames
    frames_by_color_coding, fps, num_frames = read_videos(
        recording_dir, num_color_codings
    )

    # Load kinematic states history
    kinematic_states_path = recording_dir / "kinematic_states_history.pkl"
    kinematic_states_df = pd.read_pickle(kinematic_states_path)
    timestep = kinematic_states_df["time"].iloc[1] - kinematic_states_df["time"].iloc[0]
    print(f"Loaded kinematic states with timestep {timestep:.3f} seconds.")

    # Select partial recording if needed
    if len(kinematic_states_df) != num_frames:  # Check for consistency
        raise ValueError(
            f"Number of frames in video ({num_frames}) does not match "
            f"number of kinematic states ({len(kinematic_states_df)})."
        )
    if end_frame == -1:
        end_frame = len(kinematic_states_df)
    kinematic_states_df = kinematic_states_df.iloc[start_frame:end_frame]
    for i, frames in enumerate(frames_by_color_coding):
        frames_by_color_coding[i] = frames[start_frame:end_frame]

    # Detect subsegments by extracting time series of cardinal vector pointing up
    upward_cardinal_vectors = np.array(
        kinematic_states_df["cardinal_vector_up"].tolist()
    )
    subsegments_boundaries = select_subsegments(
        upward_cardinal_vectors,
        max_tilt_angle_deg,
        mask_morph_closing_size_sec,
        min_subsegment_duration_sec,
        timestep=timestep,
    )
    print(
        f"Found {len(subsegments_boundaries)} subsegments in which the fly is upright."
    )

    # Process each subsegment
    for i, (start, end) in enumerate(subsegments_boundaries):
        print(
            f"Processing subsegment {i + 1}/{len(subsegments_boundaries)} "
            f"(frames {start}:{end})"
        )
        subsegment_frames_by_color_coding = [
            frames[start:end] for frames in frames_by_color_coding
        ]
        subsegment_kinematic_states = kinematic_states_df.iloc[start:end]

        # Process the subsegment
        output_dir = recording_dir / f"subsegment_{i:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        # processed_kinematic_states_path = output_dir / "processed_kinematic_states.pkl"
        # processed_video_dir = output_dir  # save videos directly under subsegment dir
        process_subsegment(
            subsegment_frames_by_color_coding,
            subsegment_kinematic_states,
            output_dir,
            fps,
            image_crop_size,
            n_jobs=n_jobs,
        )

        # Visualize the subsegment if requested
        if visualize:
            visualize_subsegment(
                processed_subsegment_dir=output_dir,
                fps=fps,
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
