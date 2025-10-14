import numpy as np
import pandas as pd
import yaml
import imageio
from pathlib import Path
from tqdm import tqdm

from spotlight_tools.calibration import SpotlightPositionMapper
from spotlight_tools.common.dataloader import (
    load_behavior_frame,
    get_behavior_video_capture,
)

from poseforge.spotlight.spotlight_frame_extraction import (
    rotate_points_to_align,
    rotate_image_around_point,
    crop_image_and_keypoints,
)


def process_trial(
    recording_dir: Path,
    output_dir: Path,
    edge_tolerance_mm: float = 4.0,
    crop_dim: int = 900,
    crop_x_offset: int = 0,
    crop_y_offset: int = 0,
    output_jpeg_quality: int = 95,
):
    """
    Processes a single experimental trial by extracting, aligning, and
    cropping frames from the behavior video, and saving the processed
    images to the output directory.

    Args:
        recording_dir (Path): Path to the recording data.
        output_dir (Path): Path to the directory where extracted images
            will be saved.
        edge_tolerance_mm (float): Minimum distance from the arena edge to
            the keypoint of the fly closest to the edge (usually a claw)
            for a frame to be considered usable.
        crop_dim (int): Size (in pixels) of the square crop around the
            thorax.
        crop_x_offset (int): Horizontal offset for cropping.
        crop_y_offset (int): Vertical offset for cropping.
        output_jpeg_quality (int): Quality of the saved JPEG files (1-100).
    """
    # Define paths to processed data and metadata
    processed_dir = recording_dir / "processed"
    metadata_dir = recording_dir / "metadata"
    behavior_frames_metadata_path = processed_dir / "behavior_frames_metadata.csv"
    behavior_video_path = processed_dir / "behavior_video.mkv"
    pose_2d_path = processed_dir / "pose_2d/pose_2d.npz"
    behavior_calibration_params_path = (
        metadata_dir / "calibration_parameters_behavior.yaml"
    )
    recorder_config_path = metadata_dir / "recorder_config.yaml"

    # Load metadata and pose data
    behavior_frames_metadata = pd.read_csv(behavior_frames_metadata_path)
    pose_2d = np.load(pose_2d_path)
    nodes_xy = pose_2d["nodes_xy"]
    node_names = pose_2d["node_names"]
    num_frames, num_nodes, _ = nodes_xy.shape

    # Load arena size from config
    with open(recorder_config_path, "r") as f:
        recorder_config = yaml.safe_load(f)
    arena_size = (
        recorder_config["arena"]["size_x_mm"],
        recorder_config["arena"]["size_y_mm"],
    )

    # Initialize position mapper for calibration
    mapper = SpotlightPositionMapper(behavior_calibration_params_path)

    # Compute physical positions of keypoints for each frame
    stage_positions = np.repeat(
        np.expand_dims(
            behavior_frames_metadata[["x_pos_mm_interp", "y_pos_mm_interp"]].values,
            axis=1,
        ),
        repeats=num_nodes,
        axis=1,
    )
    physical_keypoint_pos = mapper.stage_and_pixel_to_physical(
        stage_positions, pose_2d["nodes_xy"]
    )

    # Determine which frames are usable (not too close to arena edge and finite)
    xmin_allowd = edge_tolerance_mm
    xmax_allowd = arena_size[0] - edge_tolerance_mm
    ymin_allowd = edge_tolerance_mm
    ymax_allowd = arena_size[1] - edge_tolerance_mm
    usable_mask = (
        np.isfinite(physical_keypoint_pos).all(axis=(1, 2))
        & (physical_keypoint_pos[:, :, 0] >= xmin_allowd).all(axis=1)
        & (physical_keypoint_pos[:, :, 0] <= xmax_allowd).all(axis=1)
        & (physical_keypoint_pos[:, :, 1] >= ymin_allowd).all(axis=1)
        & (physical_keypoint_pos[:, :, 1] <= ymax_allowd).all(axis=1)
    )
    usable_frame_ids = np.where(usable_mask)[0]
    usable_fraction = np.sum(usable_mask) / num_frames
    print(
        f"Usable frames: {np.sum(usable_mask)}/{num_frames} "
        f"({usable_fraction * 100:.2f}%)"
    )

    # Open behavior video for frame extraction
    behavior_video_capture = get_behavior_video_capture(behavior_video_path)
    thorax_index = list(node_names).index("Th")
    neck_index = list(node_names).index("N")

    successful_frame_ids = []
    for i, frame_id in tqdm(
        enumerate(usable_frame_ids),
        total=len(usable_frame_ids),
        desc="Extracting frames",
        disable=None,
    ):
        # Load original frame and keypoints
        original_frame = load_behavior_frame(
            recording_dir, frame_id, behavior_video_capture
        )
        original_keypoints = nodes_xy[frame_id, :, :]
        original_thorax_pt = original_keypoints[thorax_index, :]
        original_neck_pt = original_keypoints[neck_index, :]

        # Rotate keypoints and image to align neck-thorax axis
        rotated_keypoints, rot_angle_rad = rotate_points_to_align(
            original_keypoints,
            original_neck_pt,
            original_thorax_pt,
        )
        rotated_frame = rotate_image_around_point(
            original_frame, original_thorax_pt, rot_angle_rad
        )
        rotated_thorax_pt = rotated_keypoints[thorax_index, :]

        # Crop image and keypoints around thorax
        crop_output = crop_image_and_keypoints(
            rotated_frame,
            rotated_keypoints,
            rotated_thorax_pt,
            crop_dim,
            crop_x_offset,
            crop_y_offset,
        )
        if crop_output is None:
            # Skip frames that cannot be cropped (e.g., out of bounds)
            continue
        cropped_frame, cropped_keypoints = crop_output

        # Save cropped frame to output directory
        successful_frame_ids.append(frame_id)
        output_frame_path = output_dir / f"frame_{frame_id:09d}.jpg"
        imageio.imwrite(
            output_frame_path,
            cropped_frame,
            quality=output_jpeg_quality,
            subrectangles=True,
        )

    num_out_of_bound_frames = len(usable_frame_ids) - len(successful_frame_ids)
    print(
        f"Successfully processed {len(successful_frame_ids)} frames "
        f"({num_out_of_bound_frames} out of bounds)"
    )


if __name__ == "__main__":
    # Find all recording directories
    spotlight_data_dir = Path("bulk_data/behavior_images/spotlight")
    recording_directories = sorted(list(spotlight_data_dir.glob("20250613-fly1b-*")))
    output_basedir = Path("bulk_data/behavior_images/spotlight_aligned_and_cropped")

    # Set processing parameters
    edge_tolerance_mm = 4.0
    crop_dim = 900
    crop_x_offset = 0
    crop_y_offset = 0

    # Process each trial
    for i, recording_dir in enumerate(recording_directories):
        print(f"Processing trial {i + 1}/{len(recording_directories)}: {recording_dir}")
        output_dir = output_basedir / recording_dir.name / "all"
        if output_dir.exists():
            print(f"Output directory {output_dir} already exists, skipping.")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)

        process_trial(
            recording_dir,
            output_dir,
            edge_tolerance_mm=edge_tolerance_mm,
            crop_dim=crop_dim,
            crop_x_offset=crop_x_offset,
            crop_y_offset=crop_y_offset,
        )
