"""Shared logic for finding and summarizing continuous accepted-frame periods.

Used by `scripts/extract_continuous_periods_from_npz.py` (finds the periods
and writes the periods `.h5`) and, for the summary figure, by that same
script as well as other scripts that want to compare period statistics
(e.g. before/after an extra filtering step) without going through an `.h5`
file.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import ndimage

SERIES_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
THRESHOLD_COLOR = "#937860"


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


def period_stats_from_ranges(
    period_ranges_per_trial: dict[str, list[tuple[int, int]]],
) -> tuple[list[int], np.ndarray]:
    """Compute period-count/length statistics from in-memory period ranges.

    Args:
        period_ranges_per_trial: Maps a trial key to that trial's list of
            `(start, end)` frame index pairs (`end` exclusive).

    Returns:
        period_counts_per_trial: Number of periods for each trial.
        period_lengths: `(n_periods,)` length, in frames, of every period.
    """
    period_counts_per_trial = [
        len(ranges) for ranges in period_ranges_per_trial.values()
    ]
    period_lengths = [
        end - start
        for ranges in period_ranges_per_trial.values()
        for start, end in ranges
    ]
    return period_counts_per_trial, np.array(period_lengths, dtype=int)


def collect_period_stats(
    periods_path_or_h5_group: Path | h5py.Group,
) -> tuple[list[int], np.ndarray]:
    """Collect per-trial period counts and all period lengths from periods data.

    Args:
        periods_path_or_h5_group: Either a periods `.h5` path (see
            `extract_continuous_periods_from_npz.py`), in which case the whole
            file is walked, or an already-open `h5py.Group` structured the
            same way (`<genotype>/<fly_trial>/<period_id>`, with `start_idx`
            and `end_idx` attrs on each period group).

    Returns:
        period_counts_per_trial: Number of periods in each `<genotype>/<fly_trial>`.
        period_lengths: `(n_periods,)` length, in frames, of every period.
    """
    if isinstance(periods_path_or_h5_group, h5py.Group):
        return _collect_period_stats_from_group(periods_path_or_h5_group)
    with h5py.File(periods_path_or_h5_group, "r") as f:
        return _collect_period_stats_from_group(f)


def _collect_period_stats_from_group(root: h5py.Group) -> tuple[list[int], np.ndarray]:
    period_ranges_per_trial = {}
    for genotype in root:
        for fly_trial in root[genotype]:
            trial_group = root[genotype][fly_trial]
            period_ranges_per_trial[f"{genotype}/{fly_trial}"] = [
                (
                    int(trial_group[period_id].attrs["start_idx"]),
                    int(trial_group[period_id].attrs["end_idx"]),
                )
                for period_id in trial_group
            ]
    return period_stats_from_ranges(period_ranges_per_trial)


def plot_period_summary(
    series: dict[str, tuple[list[int], np.ndarray]],
    min_period_length: int,
    max_length_frames: int = 700,
) -> plt.Figure:
    """Build the two-panel period summary figure.

    Args:
        series: Maps a short label (e.g. `"original"`, `"re-filtered"`) to
            that series' `(period_counts_per_trial, period_lengths)` tuple,
            as returned by `collect_period_stats` or `period_stats_from_ranges`.
            One or more entries; more than one overlays the series on the
            same axes with a legend, e.g. to compare distributions before and
            after an extra filtering step.
        min_period_length: The `min_period_length` threshold used to select
            periods (marked on the length panel), in frames.
        max_length_frames: Length panel's x-axis is clipped to
            `[0, max_length_frames]`; the KDE itself is still computed over
            each series' full data, so long-tail periods still shape the
            curve, only the display range is clipped.

    Returns:
        The figure.
    """
    fig, (ax_counts, ax_lengths) = plt.subplots(1, 2, figsize=(12, 4.5))

    totals = [
        f"{label}: {len(lengths)} periods, {int(lengths.sum())} frames total"
        for label, (_, lengths) in series.items()
    ]
    fig.suptitle("\n".join(totals) if len(totals) > 1 else totals[0])

    for (label, (counts_per_trial, _)), color in zip(series.items(), SERIES_COLORS):
        sns.kdeplot(
            x=counts_per_trial,
            fill=True,
            alpha=0.3,
            color=color,
            ax=ax_counts,
            label=label,
        )
    ax_counts.set_xlabel("Number of good periods in a trial")
    ax_counts.set_ylabel("Density (over trials)")
    ax_counts.set_title("Good periods per trial")
    ax_counts.legend(frameon=False)
    ax_counts.spines[["top", "right"]].set_visible(False)

    for (label, (_, lengths)), color in zip(series.items(), SERIES_COLORS):
        sns.kdeplot(
            x=lengths, fill=True, alpha=0.3, color=color, ax=ax_lengths, label=label
        )
    ax_lengths.axvline(
        min_period_length,
        color=THRESHOLD_COLOR,
        linestyle="--",
        linewidth=1.5,
        label=f"min_period_length ({min_period_length} frames)",
    )
    ax_lengths.set_xlim(0, max_length_frames)
    ax_lengths.set_xlabel("Period length (frames)")
    ax_lengths.set_ylabel("Density (over periods)")
    ax_lengths.set_title("Good period length distribution")
    ax_lengths.legend(frameon=False)
    ax_lengths.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
