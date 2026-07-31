#!/usr/bin/env python
"""Fit inverse kinematics (QuickIK) to a periods `.h5`'s `pred_2d_mm` data.

For each period in a periods `.h5` (see `extract_continuous_periods_from_h5.py`),
fits the NeuroMechFly body plan (see
`flygym/scripts/export_model_for_quickik.py`) to the 2D (mm) keypoint
sequence via QuickIK's `SequenceSolver` with an `XYView` mapper (see
https://github.com/NeLy-EPFL/quickik/blob/main/docs/getting-started/2d-keypoints.md),
using the same solver parameters as QuickIK's own NeuroMechFly example
(`benchmark/plot/render_video_2d.py`). Trials are solved in parallel via
joblib (one job per trial).

Only the 31 SLEAP nodes with a corresponding body-plan joint (30 leg nodes
plus "Th" -> "thorax") are used as observations; the other 6 (N, A, LA, RA,
LW, RW) have no body-plan joint and are ignored.

After solving, `ik_dofangles_rad`, `fk_3d_mm`, and `fk_2d_px` are each
independently median-filtered over time (window `--filtering-mask-frames`,
over the whole period); independently filtering these three can leave them
slightly inconsistent with each other (e.g. `fk_2d_px` no longer an exact
projection of `fk_3d_mm`), which is expected and considered acceptable here.
Only then is each frame's fit checked against its own observation: the xy
distance between `pred_2d_mm` and the (filtered) `fk_3d_mm` (dropping z) is
computed per keypoint, and a frame is rejected if its worst keypoint exceeds
`--max-mismatch` (mm) -- filtering before this check, rather than after,
means the threshold actually bounds the quality of what ends up stored.
This refined accepted mask is then run through the same
morphological-closing + minimum-length logic as
`extract_continuous_periods_from_h5.py` (closing size and minimum length
given by `--filtering-mask-frames` and `--min-period-length`), so an input
period can end up dropped entirely, shortened, or split into several
shorter periods.

Saves a full copy of the input `.h5`'s raw datasets, re-segmented into the
new periods, with three new datasets added under each:

- `ik_dofangles_rad`: `(period_length, n_dofs)` solved joint angles.
- `fk_3d_mm`: `(period_length, n_sleap_nodes, 3)` forward-kinematics result
  at the converged pose, in the same physical (mm) frame as `pred_2d_mm`,
  and in full SLEAP node order (NaN for the 6 nodes with no body-plan
  joint).
- `fk_2d_px`: Same as `fk_3d_mm`, but mapped back to the aligned-video pixel
  domain (see `poseforge.prior2d.calibration.convert_mm_to_px`), and 2D.

A summary figure comparing the input periods to the re-segmented output
periods is always saved alongside the output `.h5`.

Usage:
    python solve_ik.py \
        --periods-path lm_ported_v000_trained000_filtered_periods.h5 \
        --output-path lm_ported_v000_trained000_filtered_periods_ikfk.h5 \
        --periods-per-trial 1
"""

import json
from pathlib import Path

import h5py
import numpy as np
import quickik
import tyro
from joblib import Parallel, delayed
from loguru import logger
from scipy import ndimage

from poseforge.prior2d.calibration import (
    convert_mm_to_px,
    load_calibration_mapper,
    load_stage_positions_mm,
    load_transform_matrices,
)
from poseforge.prior2d.periods import (
    find_periods,
    period_stats_from_ranges,
    plot_period_summary,
)
from poseforge.util import configure_matplotlib_style

DATA_ROOT = Path("/mnt/upramdya_data/VAS/poseforge_paper_data")
BODY_PLAN_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "neuromechfly_ypr_legs.json"
)
# QuickIK's own NeuroMechFly example (benchmark/plot/render_video_2d.py) uses
# 10x SolverConfig's own default (1e-3), i.e. 0.01. This was raised to 0.1,
# then, after comparing 0.1/0.5/1.0 on rendered FlyGym replays, settled at
# 0.5 (this session, 2026-07-31): stronger than the QuickIK-derived starting
# point, but 1.0 over-smoothed genuine leg motion.
NEUTRAL_WEIGHT = 0.5

# ThC (leg-base) position is barely observed independently of the root pose
# it's rigidly close to (see `solve_period_ik`'s root-Missing note), so its
# observation is trusted less than the other, more informative leg keypoints.
THC_WEIGHT_SCALER = 0.5

RAW_DATASET_NAMES = ["pred_2d_px", "pred_2d_mm", "confidence"]
IK_DATASET_NAMES = ["ik_dofangles_rad", "fk_3d_mm", "fk_2d_px"]

LEG_PREFIX_MAP = {
    "LF": "lf",
    "LM": "lm",
    "LH": "lh",
    "RF": "rf",
    "RM": "rm",
    "RH": "rh",
}
LEG_JOINT_MAP = {
    "ThC": "thorax_coxa",
    "CTr": "coxa_trochanterfemur",
    "FTi": "trochanterfemur_tibia",
    "TiTa": "tibia_tarsus",
    "Cl": "claw",
}


def build_sleap_to_joint_name_map() -> dict[str, str]:
    """Map SLEAP node names to NeuroMechFly body-plan joint names.

    Returns:
        Dict from SLEAP node name to body-plan joint name, for every SLEAP
        node that has a corresponding joint (30 leg nodes plus "Th" ->
        "thorax"). The other 6 SLEAP nodes (N, A, LA, RA, LW, RW) have no
        corresponding joint and are absent from the returned dict.
    """
    mapping = {"Th": "thorax"}
    for sleap_prefix, joint_prefix in LEG_PREFIX_MAP.items():
        for sleap_joint, joint_suffix in LEG_JOINT_MAP.items():
            mapping[f"{sleap_prefix}_{sleap_joint}"] = f"{joint_prefix}_{joint_suffix}"
    return mapping


def load_body_plan(body_plan_path: Path) -> tuple[quickik.KinematicTree, list, list]:
    """Load the body-plan JSON as a `KinematicTree`, plus its name orders.

    Args:
        body_plan_path: Path to the body-plan JSON (see
            `flygym/scripts/export_model_for_quickik.py`).

    Returns:
        tree: The kinematic tree.
        joint_names: Joint names, in `tree`'s own joint order.
        dof_names: DOF names, in `State.dof_angles` order.
    """
    with open(body_plan_path) as f:
        body_plan = json.load(f)
    joint_names = [joint["name"] for joint in body_plan["joints"]]
    dof_names = [dof["name"] for joint in body_plan["joints"] for dof in joint["dofs"]]
    tree = quickik.KinematicTree.from_json_file(str(body_plan_path))
    return tree, joint_names, dof_names


def reorder_to_joint_order(
    sleap_points: np.ndarray, sleap_node_names: list[str], joint_names: list[str]
) -> np.ndarray:
    """Reorder SLEAP-order keypoints into `KinematicTree` joint order.

    Args:
        sleap_points: `(..., n_sleap_nodes, d)` array, in `sleap_node_names`
            order.
        sleap_node_names: SLEAP node names, matching `sleap_points`'s node axis.
        joint_names: Body-plan joint names, in `KinematicTree` order.

    Returns:
        `(..., len(joint_names), d)` array.
    """
    joint_to_sleap = {v: k for k, v in build_sleap_to_joint_name_map().items()}
    sleap_idx = {name: i for i, name in enumerate(sleap_node_names)}
    joint_idxs = [sleap_idx[joint_to_sleap[name]] for name in joint_names]
    return sleap_points[..., joint_idxs, :]


def reorder_from_joint_order(
    joint_points: np.ndarray, joint_names: list[str], sleap_node_names: list[str]
) -> np.ndarray:
    """Inverse of `reorder_to_joint_order`.

    Scatters joint-order keypoints back into full SLEAP node order, filling
    the 6 SLEAP nodes with no body-plan joint with NaN.

    Args:
        joint_points: `(..., len(joint_names), d)` array, in `joint_names` order.
        joint_names: Body-plan joint names, in `KinematicTree` order.
        sleap_node_names: Full SLEAP node name order to scatter into.

    Returns:
        `(..., len(sleap_node_names), d)` array.
    """
    joint_to_sleap = {v: k for k, v in build_sleap_to_joint_name_map().items()}
    sleap_idx = {name: i for i, name in enumerate(sleap_node_names)}

    out_shape = (
        *joint_points.shape[:-2],
        len(sleap_node_names),
        joint_points.shape[-1],
    )
    out = np.full(out_shape, np.nan, dtype=joint_points.dtype)
    for i, name in enumerate(joint_names):
        out[..., sleap_idx[joint_to_sleap[name]], :] = joint_points[..., i, :]
    return out


def solve_period_ik(
    pred_2d_mm: np.ndarray,
    node_names: list[str],
    tree: quickik.KinematicTree,
    joint_names: list[str],
    neutral_weight: float = NEUTRAL_WEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit IK to one period's 2D (mm) keypoint sequence.

    Args:
        pred_2d_mm: `(period_length, n_sleap_nodes, 2)` physical (mm)
            keypoints, may contain NaN.
        node_names: SLEAP node names, matching `pred_2d_mm`'s node axis.
        tree: Body-plan kinematic tree (see `load_body_plan`).
        joint_names: See `load_body_plan`.
        neutral_weight: See `main`.

    Returns:
        dof_angles: `(period_length, n_dofs)` solved joint angles, radians.
        fk_3d_mm: `(period_length, n_sleap_nodes, 3)` forward-kinematics
            result at the converged pose, in full SLEAP node order (NaN for
            nodes with no corresponding body-plan joint).
    """
    positions = reorder_to_joint_order(pred_2d_mm, node_names, joint_names)
    missing = np.any(np.isnan(positions), axis=-1)
    weights = np.where(missing, 0.0, 1.0).astype(np.float32)
    # SLEAP's "Th" landmark and the body plan's root ("thorax") joint origin
    # are different physical points on the fly (Th is an anatomical marker;
    # the body plan's thorax origin is an internal reference for the leg
    # attachment geometry), so Th is never a valid observation for the root:
    # always mark it missing, matching QuickIK's own NeuroMechFly tutorial
    # convention, and let the leg-tip observations (through the kinematic
    # chain) drive the root pose entirely.
    weights[:, joint_names.index("thorax")] = 0.0
    thc_idxs = [
        i for i, name in enumerate(joint_names) if name.endswith("_thorax_coxa")
    ]
    weights[:, thc_idxs] *= THC_WEIGHT_SCALER
    positions = np.nan_to_num(positions, nan=0.0).astype(np.float32)

    config = quickik.SolverConfig(neutral_weight=neutral_weight)
    seq_solver = quickik.SequenceSolver(tree, config, mapper=quickik.XYView())
    states, fk_positions = seq_solver.solve_sequence_with_fk(positions, weights)

    dof_angles = np.stack([state.dof_angles for state in states]).astype(np.float32)
    fk_3d_mm = reorder_from_joint_order(
        np.asarray(fk_positions), joint_names, node_names
    ).astype(np.float32)
    return dof_angles, fk_3d_mm


def compute_mismatch_mm(
    pred_2d_mm: np.ndarray, fk_3d_mm: np.ndarray, node_names: list[str]
) -> np.ndarray:
    """Per-frame worst-keypoint xy mismatch between predictions and FK, in mm.

    Args:
        pred_2d_mm: `(period_length, n_sleap_nodes, 2)` physical (mm)
            keypoints.
        fk_3d_mm: `(period_length, n_sleap_nodes, 3)` FK result, in the same
            frame as `pred_2d_mm`; NaN for nodes with no body-plan joint.
        node_names: SLEAP node names, matching both arrays' node axis.

    Returns:
        `(period_length,)` max, over leg keypoints (i.e. excluding "Th",
        which is never a valid FK comparison point; see `solve_period_ik`,
        and excluding nodes with no body-plan joint), of the xy distance
        between `pred_2d_mm` and `fk_3d_mm`.
    """
    dist = np.linalg.norm(pred_2d_mm - fk_3d_mm[..., :2], axis=-1)
    dist[:, node_names.index("Th")] = np.nan
    return np.nanmax(dist, axis=-1)


def median_filter_over_time(x: np.ndarray, window: int) -> np.ndarray:
    """Median-filter `x` along its leading (time) axis.

    Any trailing-axes "column" (e.g. one (node, coord) pair) that is NaN for
    every frame is left untouched, since `scipy.ndimage.median_filter` isn't
    NaN-aware and could otherwise fabricate finite output from a window that
    is entirely NaN.

    Args:
        x: `(period_length, ...)` array.
        window: Median filter window size, in frames. Values `<= 1` are a
            no-op.

    Returns:
        Array of the same shape as `x`.
    """
    if window <= 1:
        return x
    size = (window,) + (1,) * (x.ndim - 1)
    filtered = ndimage.median_filter(x, size=size, mode="nearest")
    all_nan = np.isnan(x).all(axis=0, keepdims=True)
    return np.where(all_nan, x, filtered)


def process_trial(
    genotype: str,
    fly_trial: str,
    periods_input: list[dict],
    data_root: Path,
    body_plan_path: Path,
    joint_names: list[str],
    dof_names: list[str],
    max_mismatch: float,
    filtering_mask_frames: int,
    min_period_length: int,
    neutral_weight: float,
) -> tuple[str, str, list[dict]]:
    """Solve IK and re-segment every period of one trial. Runs in a joblib worker.

    Args:
        genotype: Trial's genotype.
        fly_trial: Trial's fly/trial id.
        periods_input: One dict per input period, each with `start_idx`,
            `end_idx`, `node_names`, `pred_2d_px`, `pred_2d_mm`, `confidence`
            (see `main`).
        data_root: Root directory containing this trial's calibration,
            transforms, and stage position files (read-only).
        body_plan_path: Body-plan JSON; loaded fresh here since a
            `quickik.KinematicTree` can't be pickled across joblib workers.
        joint_names: See `load_body_plan`.
        dof_names: See `load_body_plan`.
        max_mismatch: See `main`.
        filtering_mask_frames: See `main`.
        min_period_length: See `main`.
        neutral_weight: See `main`.

    Returns:
        genotype, fly_trial: Passed through, to route results back to the
            right trial group in the caller.
        new_periods: One dict per re-segmented output period, each with
            `start_idx`, `end_idx` (global, into the original video) and
            every raw/IK dataset's data, ready to write directly.
    """
    trial_dir = data_root / genotype / fly_trial
    mapper = load_calibration_mapper(trial_dir)
    transform_matrices = load_transform_matrices(trial_dir)
    stage_positions_mm = load_stage_positions_mm(trial_dir)
    tree = quickik.KinematicTree.from_json_file(str(body_plan_path))

    new_periods = []
    for period in periods_input:
        start_idx = period["start_idx"]
        end_idx = period["end_idx"]
        node_names = period["node_names"]

        dof_angles, fk_3d_mm = solve_period_ik(
            period["pred_2d_mm"], node_names, tree, joint_names, neutral_weight
        )
        fk_2d_px = convert_mm_to_px(
            fk_3d_mm[..., :2],
            transform_matrices[start_idx:end_idx],
            stage_positions_mm[start_idx:end_idx],
            mapper,
        ).astype(np.float32)

        # Median-filter BEFORE checking the mismatch criterion (not after),
        # so `max_mismatch` bounds the quality of what's actually stored:
        # filtering after the check could let a frame drift back out past
        # the threshold that supposedly gated its acceptance.
        dof_angles = median_filter_over_time(dof_angles, filtering_mask_frames)
        fk_3d_mm = median_filter_over_time(fk_3d_mm, filtering_mask_frames)
        fk_2d_px = median_filter_over_time(fk_2d_px, filtering_mask_frames)

        mismatch = compute_mismatch_mm(period["pred_2d_mm"], fk_3d_mm, node_names)
        accepted_refined = mismatch <= max_mismatch
        sub_periods = find_periods(
            accepted_refined, filtering_mask_frames, min_period_length
        )

        for local_start, local_end in sub_periods:
            sl = slice(local_start, local_end)
            new_periods.append(
                {
                    "start_idx": start_idx + local_start,
                    "end_idx": start_idx + local_end,
                    "pred_2d_px": period["pred_2d_px"][sl],
                    "pred_2d_mm": period["pred_2d_mm"][sl],
                    "confidence": period["confidence"][sl],
                    "ik_dofangles_rad": dof_angles[sl],
                    "fk_3d_mm": fk_3d_mm[sl],
                    "fk_2d_px": fk_2d_px[sl],
                }
            )

    return genotype, fly_trial, new_periods


def report_mismatch_stats(output_path: Path, node_names: list[str]) -> None:
    """Log fk-to-raw-prediction disagreement stats across a whole periods `.h5`.

    Args:
        output_path: Periods `.h5` with `pred_2d_mm` and `fk_3d_mm` per period
            (see `main`).
        node_names: SLEAP node names, matching both datasets' node axis; only
            the first 30 (leg keypoints) are used, excluding "Th" (see
            `solve_period_ik`) and the 6 non-leg nodes with no body-plan joint.
    """
    leg_names = node_names[:30]
    all_dist = []
    with h5py.File(output_path, "r") as f:
        for genotype in f:
            for fly_trial in f[genotype]:
                trial = f[genotype][fly_trial]
                for period_id in trial:
                    group = trial[period_id]
                    pred = group["pred_2d_mm"][:, :30]
                    fk = group["fk_3d_mm"][:, :30, :2]
                    all_dist.append(np.linalg.norm(pred - fk, axis=-1))

    if not all_dist:
        logger.warning("No periods to report fk-to-pred mismatch stats for.")
        return

    dist = np.concatenate(all_dist, axis=0)  # (total_frames, 30)
    frame_max = np.nanmax(dist, axis=-1)
    per_node_mean = np.nanmean(dist, axis=0)
    worst_idx = int(np.nanargmax(per_node_mean))

    logger.info(
        f"fk-to-pred mismatch (mm) over {dist.shape[0]} frames, 30 leg "
        f"keypoints (Th excluded): mean={np.nanmean(dist):.4f}, "
        f"mean of per-frame worst keypoint={np.nanmean(frame_max):.4f}, "
        f"worst keypoint on average={leg_names[worst_idx]} "
        f"(mean={per_node_mean[worst_idx]:.4f})"
    )


def main(
    periods_path: Path,
    output_path: Path,
    data_root: Path = DATA_ROOT,
    body_plan_path: Path = BODY_PLAN_PATH,
    periods_per_trial: int | None = None,
    max_mismatch: float = 0.3,
    filtering_mask_frames: int = 5,
    min_period_length: int = 30,
    n_jobs: int = -1,
    neutral_weight: float = NEUTRAL_WEIGHT,
) -> None:
    """Fit IK to every period in a periods `.h5`, saving a re-segmented copy.

    Args:
        periods_path: Periods `.h5` file (see
            `extract_continuous_periods_from_h5.py`), with a `pred_2d_mm`
            dataset per period.
        output_path: Where to save the IK/FK-augmented, re-segmented copy.
        data_root: Root directory containing each trial's calibration,
            transforms, and stage position files (read-only); needed to map
            `fk_3d_mm` back to `fk_2d_px`.
        body_plan_path: Body-plan JSON (see
            `flygym/scripts/export_model_for_quickik.py`).
        periods_per_trial: If given, only fit the first N periods of each
            trial, in existing order (for quick testing). Fits every period
            by default.
        max_mismatch: Maximum tolerated xy mismatch (mm) between `pred_2d_mm`
            and `fk_3d_mm` for a frame's worst keypoint; frames exceeding
            this are rejected (see `compute_mismatch_mm`).
        filtering_mask_frames: Structuring element size for closing the
            refined accepted mask, and window size for median-filtering
            `ik_dofangles_rad`/`fk_3d_mm`/`fk_2d_px` over time (see
            `find_periods`, `median_filter_over_time`).
        min_period_length: Minimum length, in frames, for a re-segmented
            period to be kept.
        n_jobs: Number of parallel joblib workers (one trial per job); `-1`
            uses all available cores.
        neutral_weight: `quickik.SolverConfig`'s prior weight pulling the
            solution toward the body plan's neutral pose (see
            `NEUTRAL_WEIGHT`'s comment for how the default was chosen).
    """
    if not periods_path.is_file():
        raise SystemExit(f"Input file does not exist: {periods_path}")

    configure_matplotlib_style()
    _, joint_names, dof_names = load_body_plan(body_plan_path)

    with h5py.File(periods_path, "r") as f_in:
        root_attrs = dict(f_in.attrs)
        old_period_ranges: dict[str, list[tuple[int, int]]] = {}
        work_items = []
        for genotype in f_in:
            for fly_trial in f_in[genotype]:
                trial_group = f_in[genotype][fly_trial]
                period_ids = list(trial_group.keys())
                if periods_per_trial is not None:
                    period_ids = period_ids[:periods_per_trial]

                ranges = []
                periods_input = []
                for period_id in period_ids:
                    group = trial_group[period_id]
                    start_idx = int(group.attrs["start_idx"])
                    end_idx = int(group.attrs["end_idx"])
                    ranges.append((start_idx, end_idx))
                    periods_input.append(
                        {
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                            "node_names": list(group["pred_2d_px"].attrs["node_names"]),
                            "pred_2d_px": group["pred_2d_px"][:],
                            "pred_2d_mm": group["pred_2d_mm"][:],
                            "confidence": group["confidence"][:],
                        }
                    )
                old_period_ranges[f"{genotype}/{fly_trial}"] = ranges
                if periods_input:
                    work_items.append((genotype, fly_trial, periods_input))

    node_names = list(root_attrs["node_names"])
    n_input_periods = sum(len(w[2]) for w in work_items)
    logger.info(
        f"Solving IK for {n_input_periods} periods across {len(work_items)} "
        f"trials (n_jobs={n_jobs})"
    )

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_trial)(
            genotype,
            fly_trial,
            periods_input,
            data_root,
            body_plan_path,
            joint_names,
            dof_names,
            max_mismatch,
            filtering_mask_frames,
            min_period_length,
            neutral_weight,
        )
        for genotype, fly_trial, periods_input in work_items
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_period_ranges: dict[str, list[tuple[int, int]]] = {}
    n_periods_total = 0
    with h5py.File(output_path, "w") as f_out:
        for key, value in root_attrs.items():
            f_out.attrs[key] = value
        f_out.attrs["max_mismatch_mm"] = max_mismatch
        f_out.attrs["filtering_mask_frames"] = filtering_mask_frames
        f_out.attrs["min_period_length"] = min_period_length
        f_out.attrs["neutral_weight"] = neutral_weight

        for genotype, fly_trial, new_periods in results:
            trial_key = f"{genotype}/{fly_trial}"
            out_trial_group = f_out.require_group(trial_key)
            ranges = []
            for period_id, period in enumerate(new_periods):
                group = out_trial_group.create_group(str(period_id))
                for name in RAW_DATASET_NAMES + IK_DATASET_NAMES:
                    dset = group.create_dataset(
                        name, data=period[name], compression="gzip"
                    )
                    if name == "ik_dofangles_rad":
                        dset.attrs["dof_names"] = dof_names
                    elif name != "confidence":
                        dset.attrs["node_names"] = node_names
                group.attrs["start_idx"] = period["start_idx"]
                group.attrs["end_idx"] = period["end_idx"]
                ranges.append((period["start_idx"], period["end_idx"]))
                n_periods_total += 1
            new_period_ranges[trial_key] = ranges
            logger.info(f"{trial_key}: {len(new_periods)} periods after re-filtering")

        # Trials with zero input periods have no work item (and so no entry in
        # `results`) but should still get an (empty) group, matching
        # `extract_continuous_periods_from_h5.py`'s convention of always
        # representing every trial.
        for trial_key in old_period_ranges:
            f_out.require_group(trial_key)
            new_period_ranges.setdefault(trial_key, [])

    logger.info(f"Fit IK for {n_periods_total} periods; saved to {output_path}")

    summary_path = output_path.with_name(output_path.stem + "_summary.png")
    fig = plot_period_summary(
        {
            "original": period_stats_from_ranges(old_period_ranges),
            "re-filtered": period_stats_from_ranges(new_period_ranges),
        },
        min_period_length=min_period_length,
    )
    fig.savefig(summary_path, dpi=75)
    logger.info(f"Saved summary figure to {summary_path}")

    report_mismatch_stats(output_path, node_names)


if __name__ == "__main__":
    tyro.cli(main)
