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


def _world_xyz_to_seqikpy_format(
    world_xyz: np.ndarray,
    keypoint_names_canonical: list[str] | np.ndarray,
    max_n_frames: int | None = None,
) -> dict[str, np.ndarray]:
    """Convert raw 3D keypoint positions to format expected by SeqIKPy."""
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
        assert not np.isnan(data_block).any()
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


def align_fwdkin_xyz_to_rawpred_xyz(
    keypoints_pos_raw: np.ndarray,
    keypoints_pos_constrained: np.ndarray,
    keypoints_order: list[str],
    legs: list[str],
    leg_keypoints_canonical: list[str],
) -> np.ndarray:
    """Align constrained poses to raw poses by shifting each leg's kinematic chain.

    The inverse kinematics process aligns each leg to a template position. For visualization,
    we want to shift each leg back so that the first keypoint (ThC/Coxa) has the same 3D
    position as in the raw poses.

    Args:
        keypoints_pos_raw: Raw keypoint positions (n_frames, n_keypoints, 3)
        keypoints_pos_constrained: Constrained keypoint positions (n_frames, n_keypoints, 3)
        keypoints_order: List of keypoint names
        legs: List of leg names ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
        leg_keypoints_canonical: List of keypoint names per leg ['ThC', 'CTr', 'FTi', 'TiTa', 'Claw']

    Returns:
        keypoints_pos_constrained_aligned: Aligned constrained poses
    """
    keypoints_pos_constrained_aligned = keypoints_pos_constrained.copy()
    n_frames = keypoints_pos_raw.shape[0]

    # For each leg, align the constrained pose to the raw pose
    for leg in legs:
        # Get the first keypoint (ThC/Coxa) for this leg
        first_keypoint_name = f"{leg}{leg_keypoints_canonical[0]}"  # e.g., "LFThC"

        try:
            first_keypoint_idx = keypoints_order.index(first_keypoint_name)
        except ValueError:
            logger.warning(
                f"Keypoint {first_keypoint_name} not found in keypoints_order"
            )
            continue

        # Get all keypoint indices for this leg
        leg_keypoint_indices = []
        for keypoint in leg_keypoints_canonical:
            keypoint_name = f"{leg}{keypoint}"
            try:
                idx = keypoints_order.index(keypoint_name)
                leg_keypoint_indices.append(idx)
            except ValueError:
                logger.warning(f"Keypoint {keypoint_name} not found in keypoints_order")
                continue

        if not leg_keypoint_indices:
            continue

        # For each frame, compute the translation needed to align the first keypoint
        for frame_idx in range(n_frames):
            # Get the positions of the first keypoint in raw and constrained poses
            raw_first_pos = keypoints_pos_raw[frame_idx, first_keypoint_idx]
            constrained_first_pos = keypoints_pos_constrained[
                frame_idx, first_keypoint_idx
            ]

            # Skip if either position has NaN values
            if np.isnan(raw_first_pos).any() or np.isnan(constrained_first_pos).any():
                continue

            # Compute translation vector
            translation = raw_first_pos - constrained_first_pos

            # Apply translation to all keypoints of this leg
            for leg_kp_idx in leg_keypoint_indices:
                current_pos = keypoints_pos_constrained_aligned[frame_idx, leg_kp_idx]
                if not np.isnan(current_pos).any():
                    keypoints_pos_constrained_aligned[frame_idx, leg_kp_idx] = (
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
