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
`<genotype>/<fly_trial>/<period_id_within_video>/{pred_2d_px,pred_2d_mm,confidence}`,
where `period_id_within_video` counts periods within that video from 0 in
chronological order. `pred_2d_px` is `poses` (aligned-video pixel domain) and
`pred_2d_mm` is the same keypoints converted to physical (mm) coordinates
(see `convert_px_to_mm`); both have shape `(period_length, n_nodes, 2)` and
carry the node name order in their own `.attrs`. `confidence` is
`keypoint_scores` (SLEAP's own per-keypoint confidence values; SLEAP does not
produce an "uncertainty" quantity) for that period's frame range. Each
period group's `.attrs` records `start_idx`/`end_idx` (into the original
video, `end_idx` exclusive). The file's root `.attrs` records the filtering
parameters used here and the skeleton's node names/order.

The mm conversion needs, per trial (under `--data-root`): the per-frame
raw-to-aligned affine transforms (`processed/behavior_alignment_transforms.h5`,
inverted here to go aligned -> raw), the per-frame stage position
(`processed/behavior_frames_metadata.csv`), and the camera calibration
(`metadata/calibration_parameters_behavior.yaml`, read directly out of
`metadata.zip` without extracting it).

A summary figure (period counts per trial and period length distribution)
is always saved alongside the periods `.h5`, at
`<output_path.stem>_summary.png`.

Usage:
    python extract_continuous_periods_from_npz.py \
        --input-path lm_ported_v000_trained000_filtered.npz \
        --output-path lm_ported_v000_trained000_filtered_periods.h5
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tyro
from loguru import logger

from poseforge.prior2d.calibration import (
    convert_px_to_mm,
    load_calibration_mapper,
    load_stage_positions_mm,
    load_transform_matrices,
)
from poseforge.prior2d.periods import (
    collect_period_stats,
    find_periods,
    plot_period_summary,
)
from poseforge.util import configure_matplotlib_style

DATA_ROOT = Path("/mnt/upramdya_data/VAS/poseforge_paper_data")


def main(
    input_path: Path,
    output_path: Path,
    data_root: Path = DATA_ROOT,
    closing_size: int = 5,
    min_period_length: int = 30,
) -> None:
    """Extract continuous periods of accepted predictions into an `.h5` file.

    Args:
        input_path: Multi-video `.npz` with `poses`, `keypoint_scores`,
            `accepted`, `genotypes`, `fly_trials`, `n_frames_per_video`, and
            `node_names` arrays (see `convert_slp.py --include-acceptance`).
        output_path: Where to save the periods `.h5` file.
        data_root: Root directory containing each trial's calibration,
            transforms, and stage position files (read-only).
        closing_size: See `find_periods`.
        min_period_length: See `find_periods`.
    """
    configure_matplotlib_style()

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

            if periods:
                trial_dir = data_root / genotype / fly_trial
                mapper = load_calibration_mapper(trial_dir)
                transform_matrices = load_transform_matrices(trial_dir)
                stage_positions_mm = load_stage_positions_mm(trial_dir)

            for period_id, (start, end) in enumerate(periods):
                pred_2d_px = poses[video_idx, start:end]
                pred_2d_mm = convert_px_to_mm(
                    pred_2d_px,
                    transform_matrices[start:end],
                    stage_positions_mm[start:end],
                    mapper,
                ).astype(np.float32)

                group = trial_group.create_group(str(period_id))
                for name, dataset in [
                    ("pred_2d_px", pred_2d_px),
                    ("pred_2d_mm", pred_2d_mm),
                ]:
                    dset = group.create_dataset(name, data=dataset, compression="gzip")
                    dset.attrs["node_names"] = node_names
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

    with h5py.File(output_path, "r") as f:
        period_counts_per_trial, period_lengths = collect_period_stats(f)
    fig = plot_period_summary(
        {"periods": (period_counts_per_trial, period_lengths)}, min_period_length
    )
    summary_path = output_path.with_name(output_path.stem + "_summary.png")
    fig.savefig(summary_path, dpi=75)
    plt.close(fig)

    logger.info(
        f"{len(period_counts_per_trial)} trials, {len(period_lengths)} total periods "
        f"({int(period_lengths.sum())} frames). Saved summary figure to {summary_path}"
    )


if __name__ == "__main__":
    tyro.cli(main)
