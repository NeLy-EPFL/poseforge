#!/usr/bin/env python
"""Render annotated overview videos for a sample of continuous good periods.

For each trial in a periods `.h5` file (see `select_continuous_periods.py`),
renders its `--periods-per-trial` longest periods as short videos, with the
predicted skeleton drawn over the corresponding aligned video's frames
(drawing done with OpenCV only, no matplotlib).

Usage:
    python visualize_periods.py \
        --periods-path lm_ported_v000_trained000_filtered_periods.h5 \
        --output-dir bulk_data/prior-2dinvkin/sleap/lm_ported/period_videos
"""

import shutil
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np
import tyro
from loguru import logger

from poseforge.prior2d.skeleton_viz import build_skeleton, draw_pose

DATA_ROOT = Path("/mnt/upramdya_data/VAS/poseforge_paper_data")
ALIGNED_VIDEO_RELPATH = Path("processed/aligned_behavior_video.mkv")


def select_longest_periods(
    trial_group: h5py.Group, periods_per_trial: int
) -> list[str]:
    """Pick the `periods_per_trial` longest period ids in one trial's group.

    Args:
        trial_group: The `<genotype>/<fly_trial>` group.
        periods_per_trial: Maximum number of periods to select.

    Returns:
        Period ids (group keys), longest first.
    """
    period_ids = list(trial_group.keys())
    period_ids.sort(
        key=lambda pid: (
            trial_group[pid].attrs["end_idx"] - trial_group[pid].attrs["start_idx"]
        ),
        reverse=True,
    )
    return period_ids[:periods_per_trial]


def render_period_video(
    video_path: Path,
    predictions: np.ndarray,
    start_idx: int,
    edges,
    point_colors,
    output_path: Path,
    crf: int,
) -> None:
    """Render one period's aligned-video frames with the skeleton overlaid.

    Args:
        video_path: Aligned video the period was drawn from.
        predictions: `(period_length, n_nodes, 2)` keypoint coordinates.
        start_idx: First frame index of the period within `video_path`.
        edges: See `build_skeleton`.
        point_colors: See `build_skeleton`.
        output_path: Output video path.
        crf: x264 CRF for the output video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-an", "-vcodec", "libx264", "-crf", str(crf),
        "-pix_fmt", "yuv420p", str(output_path),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for local_idx in range(predictions.shape[0]):
            ret, frame = cap.read()
            if not ret:
                logger.warning(
                    f"{video_path}: ran out of frames {start_idx + local_idx} "
                    f"frames in; period video will be shorter than expected"
                )
                break
            draw_pose(frame, predictions[local_idx], edges, point_colors)
            proc.stdin.write(frame.tobytes())
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()


def main(
    periods_path: Path,
    output_dir: Path,
    data_root: Path = DATA_ROOT,
    periods_per_trial: int = 3,
    crf: int = 18,
) -> None:
    """Render annotated overview videos for a sample of continuous good periods.

    Args:
        periods_path: Periods `.h5` file (see `select_continuous_periods.py`).
        output_dir: Directory to save the rendered period videos to.
        data_root: Root directory containing the aligned videos (read-only).
        periods_per_trial: Number of (longest) periods to render per trial.
        crf: x264 CRF for the output videos; lower is higher quality.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg not found on PATH; load/install ffmpeg to render period videos."
        )
    if not periods_path.is_file():
        raise SystemExit(f"Input file does not exist: {periods_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(periods_path, "r") as f:
        node_names = list(f.attrs["node_names"])
        edges, point_colors = build_skeleton(node_names)

        n_rendered = 0
        for genotype in f:
            for fly_trial in f[genotype]:
                video_path = data_root / genotype / fly_trial / ALIGNED_VIDEO_RELPATH
                if not video_path.is_file():
                    logger.warning(
                        f"Skipping {genotype}/{fly_trial}: video not found at {video_path}"
                    )
                    continue

                trial_group = f[genotype][fly_trial]
                period_ids = select_longest_periods(trial_group, periods_per_trial)
                for period_id in period_ids:
                    group = trial_group[period_id]
                    start_idx = int(group.attrs["start_idx"])
                    end_idx = int(group.attrs["end_idx"])
                    output_path = (
                        output_dir / f"{genotype}__{fly_trial}__period{period_id}_"
                        f"f{start_idx}-{end_idx}.mp4"
                    )
                    render_period_video(
                        video_path,
                        group["predictions"][:],
                        start_idx,
                        edges,
                        point_colors,
                        output_path,
                        crf,
                    )
                    n_rendered += 1
                logger.info(
                    f"{genotype}/{fly_trial}: rendered {len(period_ids)} period videos"
                )

    logger.info(f"Rendered {n_rendered} period videos to {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
