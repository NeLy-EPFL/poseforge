import numpy as np
import h5py
from loguru import logger
from collections import defaultdict
from pathlib import Path
from seqikpy.alignment import AlignPose
from seqikpy.kinematic_chain import KinematicChainSeq
from seqikpy.leg_inverse_kinematics import LegInvKinSeq

import poseforge.neuromechfly.constants as nmf_constants
from poseforge.pose.keypoints3d.visualizer import visualize_leg_segment_lengths


def _interpolate_nan_frames(
    data_block: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Linearly interpolate (and edge-fill) NaN values along the time axis.

    seqikpy's IK solver has no NaN handling: a single NaN/occluded keypoint that
    reaches the solver propagates to NaN joint angles, which then trips the
    ``assert not np.isnan(...)`` at IK save time and aborts the whole recording
    (issue #48, finding I3-D). Rather than aborting, we fill short occlusion gaps
    per keypoint/coordinate so the solver always receives finite input.

    The fill is per (keypoint, coordinate) time series:
      * interior NaNs are linearly interpolated from the nearest finite
        neighbours on each side, and
      * leading/trailing NaNs are filled with the nearest finite value
        (forward/backward fill at the edges).

    Args:
        data_block: array of shape (n_frames, n_keypoints, 3). Modified on a copy.

    Returns:
        (filled, n_nan_values, n_affected_frames): the filled array (a copy), the
        number of NaN scalar values present before filling, and the number of
        frames that had at least one NaN coordinate before filling. If an entire
        time series for a (keypoint, coordinate) is NaN it cannot be filled and
        remains NaN; the caller is responsible for surfacing that.
    """
    filled = data_block.astype(np.float32, copy=True)
    n_frames = filled.shape[0]
    nan_mask = np.isnan(filled)
    n_nan_values = int(nan_mask.sum())
    n_affected_frames = int(nan_mask.any(axis=(1, 2)).sum())
    if n_nan_values == 0:
        return filled, 0, 0

    frame_idx = np.arange(n_frames)
    # Flatten (keypoint, coord) into independent time series.
    flat = filled.reshape(n_frames, -1)
    flat_nan = nan_mask.reshape(n_frames, -1)
    for series_idx in range(flat.shape[1]):
        col_nan = flat_nan[:, series_idx]
        if not col_nan.any():
            continue
        valid = ~col_nan
        if not valid.any():
            # Entire series is NaN; nothing to interpolate from. Leave as NaN.
            continue
        # np.interp clamps to the first/last valid value at the edges, which
        # gives us forward/backward fill for leading/trailing NaNs for free.
        flat[col_nan, series_idx] = np.interp(
            frame_idx[col_nan], frame_idx[valid], flat[valid, series_idx]
        )
    return filled, n_nan_values, n_affected_frames


def _world_xyz_to_seqikpy_format(
    world_xyz: np.ndarray,
    keypoint_names_canonical: list[str] | np.ndarray,
    max_n_frames: int | None = None,
) -> dict[str, np.ndarray]:
    """Convert raw 3D keypoint positions to format expected by SeqIKPy.

    Occluded/NaN keypoints are not fatal: NaN gaps are interpolated per leg over
    time and a warning is logged with the count and location (issue #48, finding
    I3-D). Only a keypoint that is NaN for *every* frame of a recording is left
    as NaN and aborts (it cannot be recovered and would otherwise silently corrupt
    the whole leg chain).
    """
    n_frames, n_keypoints, _ = world_xyz.shape
    if max_n_frames is not None:
        n_frames = min(n_frames, max_n_frames)
    keypoint_names_canonical = list(keypoint_names_canonical)
    assert n_keypoints == len(keypoint_names_canonical)

    pose_data_dict = {}
    for leg in nmf_constants.legs:
        data_block = np.full(
            (n_frames, len(nmf_constants.leg_keypoints_nmf), 3),
            np.nan,
            dtype=np.float32,
        )
        for keypoint_idx, keypoint_name in enumerate(nmf_constants.leg_keypoints_nmf):
            poseforge_key = f"{leg}{nmf_constants.keypoint_name_lookup_nmf_to_canonical[keypoint_name]}"
            idx = keypoint_names_canonical.index(poseforge_key)
            data_block[:, keypoint_idx, :] = world_xyz[:n_frames, idx, :]

        # Gracefully handle NaN/occluded keypoints instead of aborting the whole
        # recording on a single NaN (was: `assert not np.isnan(data_block).any()`).
        data_block, n_nan_values, n_affected_frames = _interpolate_nan_frames(
            data_block
        )
        if n_nan_values > 0:
            logger.warning(
                f"Leg {leg}: found {n_nan_values} NaN coordinate value(s) "
                f"(occluded keypoints) in {n_affected_frames}/{n_frames} frames; "
                f"linearly interpolated over time before inverse kinematics."
            )
        # Any remaining NaN means a keypoint was occluded for the entire recording
        # and could not be recovered. This would corrupt the IK chain, so fail loud
        # and clear (per-leg) rather than producing silently wrong angles.
        if np.isnan(data_block).any():
            fully_nan = np.isnan(data_block).all(axis=0)  # (n_keypoints, 3)
            bad_kp_idxs = sorted({int(i) for i, _ in zip(*np.where(fully_nan))})
            bad_kps = [nmf_constants.leg_keypoints_nmf[i] for i in bad_kp_idxs]
            raise ValueError(
                f"Leg {leg}: keypoint(s) {bad_kps} are NaN for the entire "
                f"recording and cannot be interpolated. Cannot run inverse "
                f"kinematics for this leg."
            )
        pose_data_dict[f"{leg}_leg"] = data_block

    return pose_data_dict


def extract_leg_segment_lengths(
    pose_data_dict: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    leg_segment_lengths_over_time = defaultdict(list)
    for side in "LR":
        for pos in "FMH":
            for seg_idx, seg_name in enumerate(["Coxa", "Femur", "Tibia", "Tarsus"]):
                leg = f"{side}{pos}"
                start_positions = pose_data_dict[f"{leg}_leg"][:, seg_idx, :]
                end_positions = pose_data_dict[f"{leg}_leg"][:, seg_idx + 1, :]
                segment_vectors = end_positions - start_positions
                segment_lengths = np.linalg.norm(segment_vectors, axis=1)
                leg_segment_lengths_over_time[f"{leg}_{seg_name}"] = segment_lengths
    return leg_segment_lengths_over_time


def calculate_average_leg_segment_lengths(
    leg_segment_lengths_over_time: dict[str, np.ndarray],
    fractional_discarded_margin: float = 0.1,
    make_symmetric: bool = True,
) -> dict[str, float]:
    average_leg_segment_lengths = {}
    for segment_name, lengths in leg_segment_lengths_over_time.items():
        min_val = np.percentile(lengths, 100 * fractional_discarded_margin)
        max_val = np.percentile(lengths, 100 * (1 - fractional_discarded_margin))
        filtered_lengths = lengths[(lengths >= min_val) & (lengths <= max_val)]
        average_length = np.mean(filtered_lengths)
        average_leg_segment_lengths[segment_name] = average_length

    if make_symmetric:
        for pos in "FMH":
            for seg_name in ["Coxa", "Femur", "Tibia", "Tarsus"]:
                left_key = f"L{pos}_{seg_name}"
                right_key = f"R{pos}_{seg_name}"
                mean_length = np.mean(
                    [
                        average_leg_segment_lengths[left_key],
                        average_leg_segment_lengths[right_key],
                    ]
                )
                average_leg_segment_lengths[left_key] = mean_length
                average_leg_segment_lengths[right_key] = mean_length

    for side in "LR":
        for pos in "FMH":
            total_length = 0.0
            for seg_name in ["Coxa", "Femur", "Tibia", "Tarsus"]:
                segment_key = f"{side}{pos}_{seg_name}"
                total_length += average_leg_segment_lengths[segment_key]
            average_leg_segment_lengths[f"{side}{pos}"] = total_length

    return average_leg_segment_lengths


def scale_template_by_leg_segment_sizes(
    template_positions_dict, size_dict
) -> dict[str, np.ndarray]:
    segments = ["Coxa", "Femur", "Tibia", "Tarsus", "Claw"]
    scaled_template = {}
    for side in "LR":
        for pos in "FMH":
            leg = f"{side}{pos}"
            for seg_idx, seg_name in enumerate(segments):
                keypoint_name = f"{leg}_{seg_name}"
                pos_original = template_positions_dict[keypoint_name]
                if seg_name == "Coxa":
                    scaled_template[keypoint_name] = pos_original
                    continue
                prev_keypoint_name = f"{leg}_{segments[seg_idx - 1]}"
                prev_pos_original = template_positions_dict[prev_keypoint_name]
                size_original = np.linalg.norm(pos_original - prev_pos_original)
                size_new = size_dict[prev_keypoint_name]
                direction_vector = pos_original - prev_pos_original
                scaled_vector = direction_vector * (size_new / size_original)
                pos_new = scaled_template[prev_keypoint_name] + scaled_vector
                scaled_template[keypoint_name] = pos_new
                logger.info(
                    f"Scaled {keypoint_name}: {size_original:.3f} -> {size_new:.3f}. "
                    f"This is based on prev keypoint {prev_keypoint_name}, "
                    f"curr keypoint {keypoint_name}, "
                    f"and size for segment {prev_keypoint_name}."
                )
    return scaled_template


def run_seqikpy(
    world_xyz: np.ndarray,
    keypoint_names_canonical: list[str],
    max_n_frames: int | None = None,
    n_workers: int = 6,
    debug_plots_dir: Path | None = None,
    **run_ik_and_fk_kwargs,
) -> dict[str, np.ndarray]:
    # Convert input format to what SeqIKPy expects
    pose_data_dict = _world_xyz_to_seqikpy_format(
        world_xyz, keypoint_names_canonical, max_n_frames=max_n_frames
    )

    leg_segment_lengths_over_time = extract_leg_segment_lengths(pose_data_dict)
    sizes_from_data = calculate_average_leg_segment_lengths(
        leg_segment_lengths_over_time, make_symmetric=True
    )
    for k, v in nmf_constants.nmf_size.items():
        if k not in sizes_from_data:
            sizes_from_data[k] = v  # add antennae sizes
    template_scaled_to_data = scale_template_by_leg_segment_sizes(
        nmf_constants.nmf_template, sizes_from_data
    )
    if debug_plots_dir is not None:
        debug_plots_dir.mkdir(parents=True, exist_ok=True)
        visualize_leg_segment_lengths(
            leg_segment_lengths_over_time, sizes_from_data, output_dir=debug_plots_dir
        )

    # Align keypoints so that
    #   1. Each coxa is at the coxa position in the template
    #   2. Each body segment is scaled based on how the average size in the input
    #      compares to the size in the template
    pose_aligner = AlignPose(
        pose_data_dict,
        legs_list=nmf_constants.legs,
        include_claw=True,
        body_template=template_scaled_to_data,
        body_size=sizes_from_data,
    )
    aligned_pose = pose_aligner.align_pose()

    # Run inverse kinematics and forward kinematics.
    # This is the slowest part and it's parallelized at leg level (so use n_workers=6).
    seq_kinematic_chain = KinematicChainSeq(
        bounds_dof=nmf_constants.nmf_bounds,
        legs_list=nmf_constants.legs,
        body_size=sizes_from_data,
    )
    leg_seq_ik = LegInvKinSeq(
        aligned_pos=aligned_pose,
        kinematic_chain_class=seq_kinematic_chain,
        initial_angles=nmf_constants.nmf_initial_angles,
    )
    joint_angles, forward_kinematics = leg_seq_ik.run_ik_and_fk(
        n_workers=n_workers, **run_ik_and_fk_kwargs
    )

    return joint_angles, forward_kinematics


def detect_large_joint_angle_jumps(
    joint_angles: dict[str, np.ndarray],
    threshold_rad: float = np.deg2rad(45.0),
) -> dict[str, np.ndarray]:
    """Detect large frame-to-frame jumps in per-DOF joint-angle time series.

    seqikpy seeds the IK solve at frame ``t`` with the solution from frame
    ``t-1``, so a single bad solve can propagate; and with
    ``parallel_over_time=True`` each time chunk is re-seeded from the static
    initial angles and chunks are linearly blended, which can introduce wrong
    transients at chunk boundaries (issue #48, finding I3-C).

    This is a *post-hoc* detector: it does not fix the angles, it flags frames
    whose absolute change from the previous frame exceeds ``threshold_rad`` so a
    caller can log / inspect them. It is intentionally pure (dict of numpy arrays
    in, dict of boolean masks out) so it is easy to test and reuse.

    Args:
        joint_angles: mapping of ``Angle_{leg}_{dof}`` -> 1D array of radians.
        threshold_rad: absolute per-frame jump (radians) above which a frame is
            flagged. Defaults to 45 degrees.

    Returns:
        Mapping from each input key to a boolean mask of shape (n_frames,) that is
        ``True`` at frame ``t`` when ``|angle[t] - angle[t-1]| > threshold_rad``.
        Frame 0 is always ``False`` (no previous frame). Keys whose values are not
        1D arrays are skipped.
    """
    jumps: dict[str, np.ndarray] = {}
    for key, series in joint_angles.items():
        series = np.asarray(series)
        if series.ndim != 1 or series.shape[0] < 2:
            continue
        diff = np.abs(np.diff(series))
        mask = np.zeros(series.shape[0], dtype=bool)
        mask[1:] = diff > threshold_rad
        jumps[key] = mask
    return jumps


def log_large_joint_angle_jumps(
    joint_angles: dict[str, np.ndarray],
    threshold_rad: float = np.deg2rad(45.0),
) -> int:
    """Run :func:`detect_large_joint_angle_jumps` and log a warning summary.

    Returns the total number of flagged (DOF, frame) pairs. Logs nothing beyond a
    debug line when no jumps are found.
    """
    jumps = detect_large_joint_angle_jumps(joint_angles, threshold_rad=threshold_rad)
    total = int(sum(int(mask.sum()) for mask in jumps.values()))
    if total == 0:
        logger.debug(
            f"No frame-to-frame joint-angle jumps above "
            f"{np.rad2deg(threshold_rad):.0f} deg detected."
        )
        return 0

    # Summarise per DOF (only those with at least one jump), most jumps first.
    per_dof = {k: int(m.sum()) for k, m in jumps.items() if m.any()}
    summary = ", ".join(
        f"{k}:{n}" for k, n in sorted(per_dof.items(), key=lambda kv: -kv[1])
    )
    logger.warning(
        f"Detected {total} large frame-to-frame joint-angle jump(s) "
        f"(>{np.rad2deg(threshold_rad):.0f} deg). These can indicate a bad IK "
        f"solve propagating or a chunk-boundary transient (parallel_over_time). "
        f"Per-DOF counts: {summary}. For correctness-critical runs, consider "
        f"re-running with parallel_over_time=False (or n_workers=1)."
    )
    return total


def align_fwdkin_xyz_to_rawpred_xyz(
    keypoints_pos_raw: np.ndarray,
    keypoints_pos_constrained: np.ndarray,
    keypoints_order: list[str],
    legs: list[str],
    leg_keypoints_canonical: list[str],
    keypoints_order_constrained: list[str] | None = None,
) -> np.ndarray:
    """Align constrained poses to raw poses by shifting each leg's kinematic chain.

    The inverse kinematics process aligns each leg to a template position. For visualization,
    we want to shift each leg back so that the first keypoint (ThC/Coxa) has the same 3D
    position as in the raw poses.

    The raw and constrained arrays may use *different* keypoint orderings: the raw
    array is ordered by the inference HDF5's ``keypoints`` attribute, while the
    constrained array is produced by :func:`fwdkin_world_xyz_append_antennae`, which
    returns its own ordering. Previously this function indexed *both* arrays with the
    raw ``keypoints_order``, which silently produced wrong results if the two orders
    ever differed (issue #48, finding I3-E). We now index each array with its own
    order and validate the constrained order.

    Args:
        keypoints_pos_raw: Raw keypoint positions (n_frames, n_keypoints, 3)
        keypoints_pos_constrained: Constrained keypoint positions (n_frames, n_keypoints, 3)
        keypoints_order: List of keypoint names for ``keypoints_pos_raw``
        legs: List of leg names ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
        leg_keypoints_canonical: List of keypoint names per leg ['ThC', 'CTr', 'FTi', 'TiTa', 'Claw']
        keypoints_order_constrained: List of keypoint names for
            ``keypoints_pos_constrained``. Defaults to ``keypoints_order`` for
            backwards compatibility (i.e. the caller asserts both arrays share an
            ordering).

    Returns:
        keypoints_pos_constrained_aligned: Aligned constrained poses
    """
    if keypoints_order_constrained is None:
        keypoints_order_constrained = keypoints_order

    keypoints_pos_constrained_aligned = keypoints_pos_constrained.copy()
    n_frames = keypoints_pos_raw.shape[0]

    # For each leg, align the constrained pose to the raw pose
    for leg in legs:
        # Get the first keypoint (ThC/Coxa) for this leg, looked up separately in
        # each array's own keypoint ordering.
        first_keypoint_name = f"{leg}{leg_keypoints_canonical[0]}"  # e.g., "LFThC"

        try:
            first_keypoint_idx_raw = keypoints_order.index(first_keypoint_name)
        except ValueError:
            logger.warning(
                f"Keypoint {first_keypoint_name} not found in raw keypoints_order"
            )
            continue
        try:
            first_keypoint_idx_constr = keypoints_order_constrained.index(
                first_keypoint_name
            )
        except ValueError:
            logger.warning(
                f"Keypoint {first_keypoint_name} not found in "
                f"keypoints_order_constrained"
            )
            continue

        # Get all keypoint indices for this leg, in both orderings. We keep them
        # paired so the translation applies to the same physical keypoint in each
        # array even if the orderings differ.
        leg_keypoint_index_pairs: list[tuple[int, int]] = []
        for keypoint in leg_keypoints_canonical:
            keypoint_name = f"{leg}{keypoint}"
            try:
                idx_raw = keypoints_order.index(keypoint_name)
                idx_constr = keypoints_order_constrained.index(keypoint_name)
            except ValueError:
                logger.warning(
                    f"Keypoint {keypoint_name} not found in raw/constrained "
                    f"keypoints_order"
                )
                continue
            leg_keypoint_index_pairs.append((idx_raw, idx_constr))

        if not leg_keypoint_index_pairs:
            continue

        # For each frame, compute the translation needed to align the first keypoint
        for frame_idx in range(n_frames):
            # Get the positions of the first keypoint in raw and constrained poses
            raw_first_pos = keypoints_pos_raw[frame_idx, first_keypoint_idx_raw]
            constrained_first_pos = keypoints_pos_constrained[
                frame_idx, first_keypoint_idx_constr
            ]

            # Skip if either position has NaN values
            if np.isnan(raw_first_pos).any() or np.isnan(constrained_first_pos).any():
                continue

            # Compute translation vector
            translation = raw_first_pos - constrained_first_pos

            # Apply translation to all keypoints of this leg (indexed in the
            # constrained array's own ordering).
            for _, leg_kp_idx_constr in leg_keypoint_index_pairs:
                current_pos = keypoints_pos_constrained_aligned[
                    frame_idx, leg_kp_idx_constr
                ]
                if not np.isnan(current_pos).any():
                    keypoints_pos_constrained_aligned[frame_idx, leg_kp_idx_constr] = (
                        current_pos + translation
                    )

    return keypoints_pos_constrained_aligned


def fwdkin_world_xyz_append_antennae(
    fwdkin_world_xyz: np.ndarray,
    rawpred_world_xyz: np.ndarray,
    legs: list[str],
    leg_keypoints_canonical: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Convert forward kinematics data to canonical format matching original keypoints.
    Append antenna keypoints using xyz positions before inverse kinematics.

    Args:
        fwdkin_world_xyz: Shape (n_frames, 6, 5, 3) - 6 legs, 5 keypoints per leg
        rawpred_world_xyz: Raw predicted world xyz positions before inverse kinematics
        legs: List of leg names ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
        leg_keypoints_canonical: List of keypoint names per leg ['ThC', 'CTr', 'FTi', 'TiTa', 'Claw']

    Returns:
        keypoints_pos: Shape (n_frames, n_keypoints, 3) where n_keypoints = 6*5 + 2 (antennae)
        keypoints_order: List of keypoint names in canonical format
    """
    n_frames, n_legs, n_keypoints_per_leg, _ = fwdkin_world_xyz.shape

    # Create keypoint names in canonical format for legs
    keypoints_order = []
    for leg in legs:
        for keypoint in leg_keypoints_canonical:
            keypoints_order.append(f"{leg}{keypoint}")

    # Add antenna keypoints (these won't be present in forward kinematics but needed for consistency)
    keypoints_order.extend(["LPedicel", "RPedicel"])

    # Reshape forward kinematics to (n_frames, n_keypoints, 3)
    n_leg_keypoints = n_legs * n_keypoints_per_leg
    fwdkin_world_xyz_canonical = np.full(
        (n_frames, len(keypoints_order), 3), np.nan, dtype=np.float32
    )

    # Fill in leg keypoints
    leg_keypoints_flat = fwdkin_world_xyz.reshape(n_frames, n_leg_keypoints, 3)
    fwdkin_world_xyz_canonical[:, :n_leg_keypoints, :] = leg_keypoints_flat

    # Append antenna keypoints from raw predictions
    antenna_indices = [keypoints_order.index(x) for x in ["LPedicel", "RPedicel"]]
    antenna_world_xyz = rawpred_world_xyz[:, antenna_indices, :]
    fwdkin_world_xyz_canonical[:, antenna_indices, :] = antenna_world_xyz

    return fwdkin_world_xyz_canonical, keypoints_order
