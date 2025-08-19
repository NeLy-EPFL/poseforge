import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import imageio.v2 as imageio
import shutil
import tempfile
import os
from pathlib import Path
from tqdm import tqdm, trange

from biomechpose.simulate_nmf.utils import (
    keypoint_name_lookup_neuromechfly_to_canonical,
    leg_colors,
)


def load_video_frames(video_path: Path) -> list[np.ndarray]:
    """Load video frames from a video file."""
    if not video_path.is_file():
        raise FileNotFoundError(f"{video_path} is not a file.")

    # Use imageio to read video
    reader = imageio.get_reader(str(video_path))
    frames = []
    num_frames = reader.count_frames()
    for frame in tqdm(reader, total=num_frames, desc="Loading frames"):
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


def extract_keypoint_positions(df_row: pd.Series):
    rows, cols, depths = [], [], []
    xs, ys, zs = [], [], []
    for leg in leg_colors.keys():
        for kpt in keypoint_name_lookup_neuromechfly_to_canonical.values():
            key = f"{leg}{kpt}"
            rows.append(df_row[f"keypoint_pos_cam_{key}_row"])
            cols.append(df_row[f"keypoint_pos_cam_{key}_col"])
            depths.append(df_row[f"keypoint_pos_cam_{key}_depth"])
            xs.append(df_row[f"keypoint_pos_world_{key}_x"])
            ys.append(df_row[f"keypoint_pos_world_{key}_y"])
            zs.append(df_row[f"keypoint_pos_world_{key}_z"])
    keypoint_positions_cam = np.array([cols, rows, depths]).T
    keypoint_positions_world = np.array([xs, ys, zs]).T
    return keypoint_positions_cam, keypoint_positions_world


def process_single_frame(image: np.ndarray, df_row: pd.Series, crop_size: int):
    # Gather 2d keypoint positions
    keypoints_pos_cam, keypoints_pos_world = extract_keypoint_positions(df_row)

    # Get rotation angle and rotation matrix
    rotation_angle, rotation_matrix = get_rotation_angle_and_matrix(
        df_row["cardinal_vector_forward"]
    )

    # Rotate and center-crop image
    rotated_image = rotate_image(image, -rotation_angle)  # rotation dir is opposite
    image_transformed, start_col, start_row = center_square_crop_image(
        rotated_image, crop_size
    )

    # Rotate and transform keypoint positions in camera coordinates
    height, width = image.shape[:2]
    pos_cam_transformed = []
    for i in range(keypoints_pos_cam.shape[0]):
        col_row_vec = keypoints_pos_cam[i, :2]
        depth = keypoints_pos_cam[i, 2]
        col_row_vec /= depth  # normalize by depth
        col_row_vec -= np.array([width / 2, height / 2])  # center the coordinates
        col_rot, row_rot = rotation_matrix[:2, :2] @ col_row_vec
        col_rot = col_rot + (width / 2) - start_col
        row_rot = row_rot + (height / 2) - start_row
        pos_cam_transformed.append([col_rot, row_rot, depth])
    pos_cam_transformed = np.array(pos_cam_transformed)

    # Rotate keypoint positions in world coordinates
    pos_world_transformed = keypoints_pos_world @ rotation_matrix.T

    return image_transformed, pos_cam_transformed, pos_world_transformed


def process_subsegment(
    frames,
    kinematic_states_df,
    output_kinematic_states_path: Path,
    output_video_path: Path,
    fps: int,
    crop_size: int = 464,
) -> None:
    if len(kinematic_states_df) != len(frames):
        raise ValueError(
            f"Number of frames in video ({len(frames)}) does not match "
            f"number of kinematic states ({len(kinematic_states_df)})."
        )

    # Process each frame
    images_all = []
    pos_cam_transformed_all, pos_world_transformed_all = [], []
    for i in trange(len(frames), desc="Processing frames"):
        image_transformed, pos_cam_transformed, pos_world_transformed = (
            process_single_frame(
                frames[i], kinematic_states_df.iloc[i], crop_size=crop_size
            )
        )
        images_all.append(image_transformed)
        pos_cam_transformed_all.append(pos_cam_transformed)
        pos_world_transformed_all.append(pos_world_transformed)

    # Write processed images to disk as a video
    with imageio.get_writer(
        str(output_video_path),
        fps=fps,
        codec="libx264",
        quality=10,  # 10 is highest for imageio, lower is lower quality
        ffmpeg_params=["-crf", "18", "-preset", "slow"],  # lower crf = higher quality
    ) as writer:
        for img in tqdm(images_all, desc="Writing frames"):
            writer.append_data(img)

    # Update dataframe of kinematic states with transformed keypoint positions
    pos_cam_transformed_all = np.array(pos_cam_transformed_all, dtype=np.float32)
    pos_world_transformed_all = np.array(pos_world_transformed_all, dtype=np.float32)
    all_keypoint_names = [
        f"{leg}{kpt}"
        for leg in leg_colors.keys()
        for kpt in keypoint_name_lookup_neuromechfly_to_canonical.values()
    ]
    for i, kpt_name in enumerate(all_keypoint_names):
        for j, dim in enumerate(["col", "row", "depth"]):
            key = f"keypoint_pos_cam_{kpt_name}_{dim}"
            kinematic_states_df.loc[:, key] = pos_cam_transformed_all[:, i, j]
        for j, dim in enumerate(["x", "y", "z"]):
            key = f"keypoint_pos_world_{kpt_name}_{dim}"
            kinematic_states_df.loc[:, key] = pos_world_transformed_all[:, i, j]

    kinematic_states_df = kinematic_states_df.drop(
        columns=[
            "camera_matrix",
            "cardinal_vector_forward",
            "cardinal_vector_left",
            "cardinal_vector_up",
        ]
    )
    kinematic_states_df.to_pickle(output_kinematic_states_path)


def visualize_subsegment(
    processed_kinematic_states_path: Path,
    processed_video_path: Path,
    output_video_path: Path,
    fps: int,
    camera_elevation: float = 30.0,
    max_abs_azimuth: float = 30.0,
    azimuth_rotation_period: float = 300.0,
) -> None:
    # Load video frames
    frames, fps = load_video_frames(processed_video_path)

    # Load kinematic states history
    kinematic_states_df = pd.read_pickle(processed_kinematic_states_path)

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

    # Set up figure for visualization
    fig = plt.figure(figsize=(12, 6))
    ax_2d = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

    # Visualize each frame
    viz_frames_paths = []
    for i in trange(len(frames), desc="Visualizing frames"):
        ax_2d.clear()
        ax_3d.clear()
        azimuth = np.cos(2 * np.pi * i / azimuth_rotation_period) * max_abs_azimuth
        ax_3d.view_init(elev=camera_elevation, azim=azimuth)

        entry = kinematic_states_df.iloc[i]

        # Plot 2D image
        ax_2d.imshow(frames[i])

        # Overlay keypoints on 2D image
        for leg, color in leg_colors.items():
            rows, cols = [], []
            for kpt in keypoint_name_lookup_neuromechfly_to_canonical.values():
                key = f"{leg}{kpt}"
                rows.append(entry[f"keypoint_pos_cam_{key}_row"])
                cols.append(entry[f"keypoint_pos_cam_{key}_col"])
            ax_2d.plot(cols, rows, color=color, linewidth=2)

        # Plot 3D keypoints
        for leg, color in leg_colors.items():
            xs, ys, zs = [], [], []
            for kpt in keypoint_name_lookup_neuromechfly_to_canonical.values():
                key = f"{leg}{kpt}"
                xs.append(entry[f"keypoint_pos_world_{key}_x"])
                ys.append(entry[f"keypoint_pos_world_{key}_y"])
                zs.append(entry[f"keypoint_pos_world_{key}_z"])
            ax_3d.plot(xs, ys, zs, marker="o", color=color, linewidth=2)

        ax_2d.set_axis_off()
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_xlim(-2, 2)
        ax_3d.set_ylim(-2, 2)
        ax_3d.set_zlim(-0.5, 2)
        ax_3d.set_aspect("equal")

        # Save the figure
        viz_frame_path = temp_dir / f"frame_{i:06d}.png"
        fig.savefig(viz_frame_path)
        viz_frames_paths.append(viz_frame_path)

    # Merge video
    with imageio.get_writer(
        str(output_video_path),
        fps=fps,
        codec="libx264",
        quality=10,  # 10 is highest for imageio, lower is lower quality
        ffmpeg_params=["-crf", "18", "-preset", "slow"],  # lower crf = higher quality
    ) as writer:
        for img in tqdm(viz_frames_paths, desc="Writing frames"):
            writer.append_data(imageio.imread(img))

    # Cleanup
    plt.close(fig)
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


def process_segment(
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
):
    if not recording_dir.is_dir():
        raise FileNotFoundError(f"{recording_dir} is not a directory.")

    # Load video frames
    video_path = recording_dir / "simulation_rendering.mp4"
    frames, fps = load_video_frames(video_path)
    print(f"Loaded {len(frames)} frames from {video_path} at {fps} FPS.")

    # Load kinematic states history
    kinematic_states_path = recording_dir / "kinematic_states_history.pkl"
    kinematic_states_df = pd.read_pickle(kinematic_states_path)
    timestep = kinematic_states_df["time"].iloc[1] - kinematic_states_df["time"].iloc[0]
    print(f"Loaded kinematic states with timestep {timestep:.3f} seconds.")

    # Select partial recording if needed
    if len(kinematic_states_df) != len(frames):
        raise ValueError(
            f"Number of frames in video ({len(frames)}) does not match "
            f"number of kinematic states ({len(kinematic_states_df)})."
        )
    if end_frame == -1:
        end_frame = len(kinematic_states_df)
    kinematic_states_df = kinematic_states_df.iloc[start_frame:end_frame]
    frames = frames[start_frame:end_frame]

    # Extract time series of cardinal vector pointing up
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
        subsegment_frames = frames[start:end]
        subsegment_kinematic_states = kinematic_states_df.iloc[start:end]

        # Process the subsegment
        output_dir = recording_dir / f"subsegment_{i:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_kinematic_states_path = output_dir / "processed_kinematic_states.pkl"
        processed_video_path = output_dir / "processed_simulation_rendering.mp4"
        process_subsegment(
            subsegment_frames,
            subsegment_kinematic_states,
            processed_kinematic_states_path,
            processed_video_path,
            fps,
            image_crop_size,
        )

        # Visualize the subsegment if requested
        if visualize:
            output_video_path = output_dir / "visualization.mp4"
            visualize_subsegment(
                processed_kinematic_states_path,
                processed_video_path,
                output_video_path,
                fps=fps,
                camera_elevation=camera_elevation,
                max_abs_azimuth=max_abs_azimuth,
                azimuth_rotation_period=azimuth_rotation_period,
            )
