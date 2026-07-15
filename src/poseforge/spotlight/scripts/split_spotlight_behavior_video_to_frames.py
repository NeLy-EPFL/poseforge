import cv2
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import trange
import argparse

# OpenCV spawns its own thread pool per process; when we fan out one process per
# recording that oversubscribes the CPU. Keep each decoder single-threaded.
cv2.setNumThreads(1)


def process_trial(
    recording_dir: Path,
    output_dir: Path,
    output_jpeg_quality: int = 95,
    show_progress: bool = True,
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

    cap = cv2.VideoCapture(str(aligned_behavior_video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {aligned_behavior_video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video: {total_frames} frames @ {fps:.2f} FPS")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        frame_count = 0
        for frameid in trange(
            total_frames,
            desc=f"Extracting {recording_dir.name}",
            disable=None if show_progress else True,
        ):
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Failed to read frame {frameid} unexpectedly. Stopping.")
                break

            output_path = output_dir / f"frame_{frame_count:09d}.jpg"
            cv2.imwrite(str(output_path), frame, cv2_jpeg_params)
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
    return frame_count


def _process_trial_worker(task):
    """Top-level worker so it can be pickled by ProcessPoolExecutor."""
    recording_dir, output_dir = task
    process_trial(recording_dir, output_dir, show_progress=False)
    return recording_dir.name

def start():
    parser = argparse.ArgumentParser(
        description="Split spotlight aligned behavior video into individual frames."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to the spotlight recording directory containing the aligned behavior video.",
        default=Path("bulk_data/behavior_images/spotlight"),
    )
    parser.add_argument(
        "glob_pattern",
        type=str,
        help="Glob pattern to match recording directories.",
        default="fly*",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of recordings to extract in parallel (one process each).",
    )
    args = parser.parse_args()

    return args.data_dir, args.glob_pattern, args.n_workers

if __name__ == "__main__":
    # Find all recording directories
    spotlight_data_dir, glob_pattern, n_workers = start()
    recording_directories = sorted(list(spotlight_data_dir.glob(glob_pattern)))

    output_basedir = spotlight_data_dir / "spotlight_aligned_and_cropped"

    # Build the list of trials that still need extracting (skip completed ones).
    tasks = []
    for recording_dir in recording_directories:
        output_dir = output_basedir / recording_dir.name / "all"
        if output_dir.exists():
            print(f"Output directory {output_dir} already exists, skipping.")
            continue
        tasks.append((recording_dir, output_dir))

    print(f"Extracting {len(tasks)} recording(s) with {n_workers} parallel worker(s).")

    if n_workers <= 1 or len(tasks) <= 1:
        for i, (recording_dir, output_dir) in enumerate(tasks):
            print(f"Processing trial {i + 1}/{len(tasks)}: {recording_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            process_trial(recording_dir, output_dir)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_trial_worker, task): task[0].name
                for task in tasks
            }
            for i, future in enumerate(as_completed(futures)):
                name = future.result()
                print(f"Finished {i + 1}/{len(tasks)}: {name}")
