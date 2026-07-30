#!/usr/bin/env python
"""Extract continuous periods of accepted predictions from a filtered `.npz`.

Takes a multi-video `.npz` produced by `convert_slp.py --slp2npz
--include-acceptance` (e.g. the output of `filter_student_predictions.sh`)
and, for each video, finds contiguous stretches of accepted frames:

1. Apply binary morphological closing to the video's `accepted` mask, with a
   structuring element of `--closing-size` frames. This bridges short gaps
   of not-accepted frames (e.g. a single frame that failed the acceptance
   criteria in the middle of an otherwise-good stretch) so they don't split
   an otherwise-continuous period.
2. Keep only the resulting contiguous accepted runs that are at least
   `--min-period-length` frames long.

Each kept period is saved to the output `.h5` at
`<genotype>/<fly_trial>/<period_id_within_video>/{predictions,confidence}`,
where `period_id_within_video` counts periods within that video from 0 in
chronological order. `predictions` is `poses` and `confidence` is
`keypoint_scores` (SLEAP's own per-keypoint confidence values; SLEAP does not
produce an "uncertainty" quantity) for that period's frame range. Each
period group's `.attrs` records `start_idx`/`end_idx` (into the original
video, `end_idx` exclusive). The file's root `.attrs` records the filtering
parameters used here and the skeleton's node names/order.

Usage:
    python select_continuous_periods.py \
        --input-path lm_ported_v000_trained000_filtered.npz \
        --output-path lm_ported_v000_trained000_filtered_periods.h5
"""

from pathlib import Path

import h5py
import numpy as np
import tyro
from loguru import logger
from scipy import ndimage


def find_periods(
    accepted: np.ndarray, closing_size: int, min_period_length: int
) -> list[tuple[int, int]]:
    """Find contiguous accepted periods in one video, after morphological closing.

    Args:
        accepted: `(n_frames,)` bool, whether each frame was accepted.
        closing_size: Length of the structuring element used for binary
            closing; bridges gaps of up to about this many consecutive
            not-accepted frames.
        min_period_length: Minimum number of frames for a period to be kept.

    Returns:
        List of `(start, end)` frame index pairs (`end` exclusive), one per
        kept period, in chronological order.
    """
    closed = ndimage.binary_closing(
        accepted, structure=np.ones(closing_size, dtype=bool)
    )
    labeled_periods, n_periods = ndimage.label(closed)

    periods = []
    for period_id in range(1, n_periods + 1):
        frame_idxs = np.flatnonzero(labeled_periods == period_id)
        start, end = int(frame_idxs[0]), int(frame_idxs[-1]) + 1
        if end - start >= min_period_length:
            periods.append((start, end))
    return periods


def main(
    input_path: Path,
    output_path: Path,
    closing_size: int = 5,
    min_period_length: int = 30,
) -> None:
    """Extract continuous periods of accepted predictions into an `.h5` file.

    Args:
        input_path: Multi-video `.npz` with `poses`, `keypoint_scores`,
            `accepted`, `genotypes`, `fly_trials`, `n_frames_per_video`, and
            `node_names` arrays (see `convert_slp.py --include-acceptance`).
        output_path: Where to save the periods `.h5` file.
        closing_size: See `find_periods`.
        min_period_length: See `find_periods`.
    """
    if not input_path.is_file():
        raise SystemExit(f"Input file does not exist: {input_path}")
    with np.load(input_path) as data:
        if "accepted" not in data.files:
            raise SystemExit(
                f"{input_path} has no 'accepted' array; convert with "
                "`convert_slp.py --slp2npz --include-acceptance` first."
            )
        poses = data["poses"]
        keypoint_scores = data["keypoint_scores"]
        accepted = data["accepted"]
        genotypes = data["genotypes"]
        fly_trials = data["fly_trials"]
        n_frames_per_video = data["n_frames_per_video"]
        node_names = [str(n) for n in data["node_names"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_periods_total = 0
    with h5py.File(output_path, "w") as f:
        f.attrs["closing_size"] = closing_size
        f.attrs["min_period_length"] = min_period_length
        f.attrs["node_names"] = node_names

        for video_idx in range(poses.shape[0]):
            n_frames = int(n_frames_per_video[video_idx])
            genotype = str(genotypes[video_idx])
            fly_trial = str(fly_trials[video_idx])

            periods = find_periods(
                accepted[video_idx, :n_frames], closing_size, min_period_length
            )
            # Create the trial group even with zero periods, so a trial with no
            # good periods is still represented (as an empty group) rather than
            # silently absent from the file.
            trial_group = f.require_group(f"{genotype}/{fly_trial}")
            for period_id, (start, end) in enumerate(periods):
                group = trial_group.create_group(str(period_id))
                group.create_dataset(
                    "predictions",
                    data=poses[video_idx, start:end],
                    compression="gzip",
                )
                group.create_dataset(
                    "confidence",
                    data=keypoint_scores[video_idx, start:end],
                    compression="gzip",
                )
                group.attrs["start_idx"] = start
                group.attrs["end_idx"] = end

            n_periods_total += len(periods)
            logger.info(f"{genotype}/{fly_trial}: {len(periods)} periods")

    logger.info(f"Saved {n_periods_total} periods to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
