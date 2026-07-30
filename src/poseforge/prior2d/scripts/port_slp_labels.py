#!/usr/bin/env python
"""Port per-trial SLEAP predictions into a single, aligned-domain .slp file.

For every trial under `--data-root` that has the required files (see below),
this script applies the per-trial affine transform to the predicted
keypoints, moving them from the raw camera domain into the aligned/cropped
video domain, and adds a `PredictedInstance` for every frame with a detected
pose. No confidence-based filtering or promotion to user labels happens here;
see `filter_slp_labels.py` for that.

All trials are bundled into one `sleap_io.Labels` object (one `Video` per
trial) and saved as a single new-format .slp file (see https://sleap.ai/).

Trials are sorted by path and restricted to `TRIAL_RANGE`, so re-running with
a different range (e.g. train/val/test) produces disjoint trial sets.

With `--aligned-input` (default False), each trial's h5 is expected to
already contain aligned-domain coordinates and no per-frame transform is
applied; `--transforms-relpath` is then unused. With the default
`--no-aligned-input`, each trial needs both `--h5-relpath` (raw camera
domain) and `--transforms-relpath` (per-frame affine matrices).

Usage:
    python port_slp_labels.py \
        --output-path bulk_data/prior-2dinvkin/sleap/lm_ported/lm_ported_v000_unfiltered.slp
"""

from pathlib import Path

import h5py
import numpy as np
import sleap_io as sio
import tyro
from loguru import logger

from poseforge.prior2d.geometry import apply_affine

H5_RELPATH = Path("sleap/prediction_lm_full_behavior_video.h5")
REFERENCE_SLP_RELPATH = Path("sleap/prediction_lm_full_behavior_video.slp")
TRANSFORMS_RELPATH = Path("processed/behavior_alignment_transforms.h5")
ALIGNED_VIDEO_RELPATH = Path("processed/aligned_behavior_video.mkv")

# Trials are sorted by path (see `find_trial_dirs`) and sliced to this
# half-open index range, giving a simple, reproducible train/val/test split:
# e.g. (0, 63) for train, (63, 68) for val, (68, 73) for test. Adjust and
# re-run with a different --output-path to generate each split.
TRIAL_RANGE = (0, 63)


def find_trial_dirs(
    data_root: Path,
    h5_relpath: Path,
    aligned_video_relpath: Path,
    transforms_relpath: Path | None,
) -> list[Path]:
    """Find trial directories with the h5, aligned video, and (if required)
    transforms files.

    Args:
        data_root: Root directory to search (searched recursively).
        h5_relpath: Trial-relative path to the predictions h5.
        aligned_video_relpath: Trial-relative path to the aligned video.
        transforms_relpath: Trial-relative path to the per-frame affine
            transforms h5, or None if the input is already aligned (no
            transform needed).

    Returns:
        Sorted list of trial directory paths.
    """
    trial_dirs = []
    for h5_path in sorted(data_root.rglob(str(h5_relpath))):
        # h5_relpath may have subdirectory components; the trial dir is the
        # ancestor `len(h5_relpath.parts)` levels up from h5_path.
        trial_dir = h5_path
        for _ in h5_relpath.parts:
            trial_dir = trial_dir.parent

        has_video = (trial_dir / aligned_video_relpath).is_file()
        has_transforms = (
            transforms_relpath is None or (trial_dir / transforms_relpath).is_file()
        )
        if has_video and has_transforms:
            trial_dirs.append(trial_dir)
        else:
            logger.warning(
                f"Skipping {trial_dir}: missing "
                f"{'aligned video' if not has_video else ''} "
                f"{'transforms' if not has_transforms else ''}".strip()
            )
    return trial_dirs


def build_predicted_frames(
    trial_dir: Path,
    skeleton: sio.Skeleton,
    aligned_input: bool,
    h5_relpath: Path,
    transforms_relpath: Path,
    aligned_video_relpath: Path,
    max_frames: int | None,
) -> tuple[sio.Video, list[sio.LabeledFrame]]:
    """Build the aligned-domain, predictions-only labeled frames for one trial.

    Args:
        trial_dir: Trial directory containing the h5, video, and (unless
            `aligned_input`) transforms.
        skeleton: Skeleton shared across all trials.
        aligned_input: If True, `poses` in the h5 are already aligned and no
            transform is applied.
        h5_relpath: Trial-relative path to the predictions h5.
        transforms_relpath: Trial-relative path to the per-frame affine
            transforms h5 (unused if `aligned_input`).
        aligned_video_relpath: Trial-relative path to the aligned video.
        max_frames: If given, only process the first `max_frames` frames
            (for quick smoke tests).

    Returns:
        video: The aligned video, linked from an absolute path.
        labeled_frames: One `LabeledFrame` per frame with a detected pose.
    """
    with h5py.File(trial_dir / h5_relpath, "r") as f:
        poses = f["poses"][:]
        instance_scores = f["instance_scores"][:]
        keypoint_scores = f["keypoint_scores"][:]

    if max_frames is not None:
        poses = poses[:max_frames]
        instance_scores = instance_scores[:max_frames]
        keypoint_scores = keypoint_scores[:max_frames]

    if aligned_input:
        aligned_poses = poses
    else:
        with h5py.File(trial_dir / transforms_relpath, "r") as f:
            transform_matrices = f["transform_matrices"][:]
        if max_frames is not None:
            transform_matrices = transform_matrices[:max_frames]
        if transform_matrices.shape[0] != poses.shape[0]:
            raise ValueError(
                f"{trial_dir}: frame count mismatch between h5 ({poses.shape[0]}) "
                f"and transforms ({transform_matrices.shape[0]})"
            )
        aligned_poses = apply_affine(poses, transform_matrices)

    video = sio.Video(filename=str((trial_dir / aligned_video_relpath).resolve()))

    labeled_frames = []
    for frame_idx in range(poses.shape[0]):
        if np.isnan(aligned_poses[frame_idx]).all():
            continue
        predicted_instance = sio.PredictedInstance.from_numpy(
            points_data=aligned_poses[frame_idx],
            skeleton=skeleton,
            point_scores=keypoint_scores[frame_idx],
            score=float(instance_scores[frame_idx]),
        )
        labeled_frames.append(
            sio.LabeledFrame(
                video=video, frame_idx=frame_idx, instances=[predicted_instance]
            )
        )

    logger.info(f"{trial_dir.name}: {len(labeled_frames)} predicted frames")
    return video, labeled_frames


def main(
    output_path: Path,
    data_root: Path = Path("/mnt/upramdya_data/VAS/poseforge_paper_data"),
    aligned_input: bool = False,
    h5_relpath: Path = H5_RELPATH,
    reference_slp_relpath: Path = REFERENCE_SLP_RELPATH,
    transforms_relpath: Path = TRANSFORMS_RELPATH,
    aligned_video_relpath: Path = ALIGNED_VIDEO_RELPATH,
    max_frames_per_trial: int | None = None,
) -> None:
    """Port per-trial SLEAP predictions into a new-format, aligned-domain .slp file.

    Args:
        output_path: Where to save the combined, predictions-only .slp file.
        data_root: Root directory to search for trials (read-only).
        aligned_input: If True, each trial's h5 is already in the aligned
            domain and no per-frame transform is applied (`transforms_relpath`
            is unused). If False (default), each trial's h5 is in the raw
            camera domain and is aligned using `transforms_relpath`.
        h5_relpath: Trial-relative path to the predictions h5.
        reference_slp_relpath: Trial-relative path to a `.slp` file to take
            the skeleton from (only the first trial's is used).
        transforms_relpath: Trial-relative path to the per-frame affine
            transforms h5. Unused if `aligned_input`.
        aligned_video_relpath: Trial-relative path to the aligned video.
        max_frames_per_trial: If given, only process the first N frames of
            each trial (for quick smoke tests).
    """
    trial_dirs = find_trial_dirs(
        data_root,
        h5_relpath,
        aligned_video_relpath,
        None if aligned_input else transforms_relpath,
    )
    logger.info(f"Found {len(trial_dirs)} trials under {data_root}")
    if not trial_dirs:
        raise SystemExit(f"No trials found under {data_root}")

    trial_dirs = trial_dirs[TRIAL_RANGE[0] : TRIAL_RANGE[1]]
    logger.info(f"Using trial range {TRIAL_RANGE}: {len(trial_dirs)} trials selected")

    reference_labels = sio.load_file(str(trial_dirs[0] / reference_slp_relpath))
    skeleton = reference_labels.skeleton

    videos = []
    all_labeled_frames = []
    for trial_dir in trial_dirs:
        video, labeled_frames = build_predicted_frames(
            trial_dir,
            skeleton,
            aligned_input,
            h5_relpath,
            transforms_relpath,
            aligned_video_relpath,
            max_frames_per_trial,
        )
        videos.append(video)
        all_labeled_frames.extend(labeled_frames)

    labels = sio.Labels(
        labeled_frames=all_labeled_frames, videos=videos, skeletons=[skeleton]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_file(labels, str(output_path))
    logger.info(
        f"Saved {len(all_labeled_frames)} predicted frames across {len(videos)} "
        f"videos to {output_path}"
    )


if __name__ == "__main__":
    tyro.cli(main)
