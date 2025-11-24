import cv2
import logging
from pathlib import Path
from tqdm import trange


def process_trial(
    recording_dir: Path,
    output_dir: Path,
    output_jpeg_quality: int = 95,
):
    logger = logging.getLogger(__name__)

    cv2_jpeg_params = [
        cv2.IMWRITE_JPEG_QUALITY,
        output_jpeg_quality,
        cv2.IMWRITE_JPEG_OPTIMIZE,
        1,
    ]

    aligned_behavior_video_path = recording_dir / "processed/aligned_behavior_video.mkv"
    if not aligned_behavior_video_path.exists():
        raise FileNotFoundError(
            f"Aligned behavior video not found at {aligned_behavior_video_path}"
        )

    cap = cv2.VideoCapture(aligned_behavior_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {aligned_behavior_video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video: {total_frames} frames @ {fps:.2f} FPS")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        frame_count = 0
        for frameid in trange(total_frames, desc="Extracting frames", disable=None):
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Failed to read frame {frameid} unexpectedly. Stopping.")
                break

            output_path = output_dir / f"frame_{frame_count:09d}.jpg"
            cv2.imwrite(output_path, frame, cv2_jpeg_params)
            frame_count += 1

        # check if this is really the end of the video
        ret, _ = cap.read()
        if ret:
            logging.warning(
                "There are more frames in the video than expected. "
                "These will be ignored."
            )
        logging.info(f"Extracted {frame_count} frames to {output_dir}")
    finally:
        cap.release()


if __name__ == "__main__":
    # Find all recording directories
    spotlight_data_dir = Path("bulk_data/behavior_images/spotlight")
    recording_directories = sorted(list(spotlight_data_dir.glob("20250613-fly1b-*")))
    output_basedir = Path("bulk_data/behavior_images/spotlight_aligned_and_cropped")

    # Set processing parameters
    edge_tolerance_mm = 4.0
    crop_dim = 900
    crop_shift_x = 0
    crop_shift_y = 0

    # Process each trial
    for i, recording_dir in enumerate(recording_directories):
        print(f"Processing trial {i + 1}/{len(recording_directories)}: {recording_dir}")
        output_dir = output_basedir / recording_dir.name / "all"
        if output_dir.exists():
            print(f"Output directory {output_dir} already exists, skipping.")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)

        process_trial(recording_dir, output_dir)
