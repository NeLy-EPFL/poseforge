#!/usr/bin/env python
"""Summarize a periods `.h5` file (see `select_continuous_periods.py`) in one figure.

Two panels:
- Left: number of good periods per trial, sorted descending (bar chart).
- Right: distribution of good period lengths, pooled across all trials
  (histogram), with the `min_period_length` threshold marked.

Usage:
    python summarize_periods.py \
        --periods-path lm_ported_v000_trained000_filtered_periods.h5 \
        --output-path period_summary.png
"""

from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tyro
from loguru import logger

from poseforge.util import configure_matplotlib_style

BAR_COLOR = "#4C72B0"
HIST_COLOR = "#4C72B0"
OVERFLOW_COLOR = "#93AFD4"
THRESHOLD_COLOR = "#C44E52"


def collect_period_stats(periods_path: Path) -> tuple[list[int], np.ndarray]:
    """Collect per-trial period counts and all period lengths from a periods `.h5`.

    Args:
        periods_path: Periods `.h5` file (see `select_continuous_periods.py`).

    Returns:
        period_counts_per_trial: Number of periods in each `<genotype>/<fly_trial>`.
        period_lengths: `(n_periods,)` length, in frames, of every period.
    """
    period_counts_per_trial = []
    period_lengths = []
    with h5py.File(periods_path, "r") as f:
        for genotype in f:
            for fly_trial in f[genotype]:
                trial_group = f[genotype][fly_trial]
                period_counts_per_trial.append(len(trial_group))
                for period_id in trial_group:
                    group = trial_group[period_id]
                    length = int(group.attrs["end_idx"] - group.attrs["start_idx"])
                    period_lengths.append(length)
    return period_counts_per_trial, np.array(period_lengths)


def plot_period_summary(
    period_counts_per_trial: list[int],
    period_lengths: np.ndarray,
    min_period_length: int,
    fps: float,
    max_length_s: float,
    n_bins: int,
) -> plt.Figure:
    """Build the two-panel summary figure.

    Args:
        period_counts_per_trial: See `collect_period_stats`.
        period_lengths: See `collect_period_stats`.
        min_period_length: The `min_period_length` threshold used to select
            periods (marked on the length histogram), in frames.
        fps: Frame rate used to convert period lengths to seconds for display.
        max_length_s: Length histogram bins cover `[0, max_length_s]`; every
            period longer than this is pooled into one final overflow bin
            (shown in a lighter shade and labeled ">max_length_s"), so a few
            very long periods don't compress the rest of the distribution.
        n_bins: Total number of bins in the length histogram, including the
            overflow bin (so `n_bins - 1` regular bins cover `[0, max_length_s]`).

    Returns:
        The figure.
    """
    fig, (ax_counts, ax_lengths) = plt.subplots(1, 2, figsize=(12, 4.5))

    total_duration = timedelta(seconds=round(period_lengths.sum() / fps))
    fig.suptitle(
        f"{len(period_lengths)} continuous periods, "
        f"{total_duration} total (at {fps:g} fps)"
    )

    sorted_counts = sorted(period_counts_per_trial, reverse=True)
    ax_counts.bar(
        np.arange(len(sorted_counts)), sorted_counts, color=BAR_COLOR, width=0.9
    )
    ax_counts.set_xlabel("Trial (sorted)")
    ax_counts.set_ylabel("Number of good periods")
    ax_counts.set_title(f"Good periods per trial (n={len(sorted_counts)} trials)")
    ax_counts.spines[["top", "right"]].set_visible(False)

    length_seconds = period_lengths / fps
    bin_edges = np.linspace(0, max_length_s, n_bins)
    bin_width = bin_edges[1] - bin_edges[0]
    counts, _ = np.histogram(length_seconds, bins=bin_edges)
    overflow_count = int((length_seconds > max_length_s).sum())

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    overflow_center = bin_edges[-1] + bin_width / 2

    ax_lengths.bar(bin_centers, counts, width=bin_width, color=HIST_COLOR)
    ax_lengths.bar(
        overflow_center, overflow_count, width=bin_width, color=OVERFLOW_COLOR
    )
    ax_lengths.axvline(
        min_period_length / fps,
        color=THRESHOLD_COLOR,
        linestyle="--",
        linewidth=1.5,
        label=f"min_period_length ({min_period_length} frames)",
    )
    # Drop any auto-tick too close to the boundary to avoid colliding with the
    # ">max_length_s" tick placed at the overflow bar's center.
    xticks = [
        t for t in ax_lengths.get_xticks() if 0 <= t <= max_length_s - bin_width
    ] + [overflow_center]
    ax_lengths.set_xticks(
        xticks, [*(f"{t:g}" for t in xticks[:-1]), f">{max_length_s:g}"]
    )
    ax_lengths.set_xlabel(f"Period length (s, at {fps:g} fps)")
    ax_lengths.set_ylabel("Number of periods")
    ax_lengths.set_title(f"Good period length distribution (n={len(period_lengths)})")
    ax_lengths.legend(frameon=False)
    ax_lengths.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main(
    periods_path: Path,
    output_path: Path,
    fps: float = 33.0,
    max_length_s: float = 20.0,
    n_bins: int = 30,
) -> None:
    """Summarize a periods `.h5` file in one figure (period counts + length distribution).

    Args:
        periods_path: Periods `.h5` file (see `select_continuous_periods.py`).
        output_path: Where to save the figure (format inferred from extension).
        fps: Frame rate used to convert period lengths to seconds for display.
        max_length_s: See `plot_period_summary`.
        n_bins: See `plot_period_summary`.
    """
    configure_matplotlib_style()

    if not periods_path.is_file():
        raise SystemExit(f"Input file does not exist: {periods_path}")

    with h5py.File(periods_path, "r") as f:
        min_period_length = int(f.attrs["min_period_length"])

    period_counts_per_trial, period_lengths = collect_period_stats(periods_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_period_summary(
        period_counts_per_trial,
        period_lengths,
        min_period_length,
        fps,
        max_length_s,
        n_bins,
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(
        f"{len(period_counts_per_trial)} trials, {len(period_lengths)} total periods "
        f"({period_lengths.sum()} frames). Saved summary figure to {output_path}"
    )


if __name__ == "__main__":
    tyro.cli(main)
