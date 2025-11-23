import numpy as np
import h5py
import logging
from collections import defaultdict
from pathlib import Path
from seqikpy.alignment import AlignPose
from seqikpy.kinematic_chain import KinematicChainSeq
from seqikpy.leg_inverse_kinematics import LegInvKinSeq

import poseforge.neuromechfly.constants as nmf_constants
from poseforge.pose.keypoints3d.visualizer import visualize_leg_segment_lengths


_logger = logging.getLogger(__name__)


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
                _logger.info(
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
    joint_angles, forward_kinematics = leg_seq_ik.run_ik_and_fk(n_workers=n_workers)

    return joint_angles, forward_kinematics


def save_seqikpy_output(
    output_path: Path | str,
    joint_angles: dict[str, np.ndarray],
    forward_kinematics: dict[str, np.ndarray],
    frame_ids: np.ndarray | list[int] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Rearrange joint angles into (frame, leg, dof) format
    dof_names_per_leg = list(nmf_constants.dof_name_lookup_canonical_to_nmf.keys())
    n_frames_joint_angles = joint_angles["Angle_LF_ThC_yaw"].shape[0]
    joint_angles_arr = np.full((n_frames_joint_angles, 6, 7), np.nan, dtype=np.float32)
    for leg_idx, leg in enumerate(nmf_constants.legs):
        for dof_idx, dof_name in enumerate(dof_names_per_leg):
            seqikpy_key = f"Angle_{leg}_{dof_name}"
            joint_angles_arr[:, leg_idx, dof_idx] = joint_angles[seqikpy_key]
    assert not np.isnan(joint_angles_arr).any()

    # Rearrange forward kinematics into (frame, leg, keypoint, 3) format
    # Note: Physical keypoints are ThC, CTr, FTi, TiTa, Claw
    #       Virtual keypoints include start and end of 0-length segments at multi-DoF
    #       joints, i.e. base ThC yaw pitch roll CTr pitch CTr roll FTi pitch TiTa pitch
    #       (see `_nmf_initial_angles`, stage 4)
    # Virtual keypoints 0, 1, 2, 3 are all at ThC  (base, yaw, pitch, roll)
    #                   4, 5 are at CTr
    #                   6 is at FTi
    #                   7 is at TiTa
    #                   8 is at Claw
    n_physical_keypoints = len(nmf_constants.leg_keypoints_nmf)
    n_frames_fwdkin, n_virtual_keypoints, _ = forward_kinematics["LF_leg"].shape
    assert n_frames_fwdkin == n_frames_joint_angles
    assert n_physical_keypoints == 5
    assert n_virtual_keypoints == 9
    fwdkin_world_xyz = np.full(
        (n_frames_fwdkin, len(nmf_constants.legs), n_virtual_keypoints, 3),
        np.nan,
        dtype=np.float32,
    )
    for leg_idx, leg in enumerate(nmf_constants.legs):
        fwdkin_world_xyz[:, leg_idx, :, :] = forward_kinematics[f"{leg}_leg"]
    assert not np.isnan(fwdkin_world_xyz).any()
    # Drop virtual keypoints that share the same physical location
    is_physical = np.array([1, 0, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    fwdkin_world_xyz = fwdkin_world_xyz[:, :, is_physical, :]

    with h5py.File(output_path, "w") as f:
        joint_angles_ds = f.create_dataset(
            "joint_angles", data=joint_angles_arr, dtype=np.float32, compression="gzip"
        )
        joint_angles_ds.attrs["legs"] = nmf_constants.legs
        joint_angles_ds.attrs["dof_names_per_leg"] = dof_names_per_leg

        fwdkin_ds = f.create_dataset(
            "forward_kinematics_world_xyz",
            data=fwdkin_world_xyz,
            dtype=np.float32,
            compression="gzip",
        )
        fwdkin_ds.attrs["legs"] = nmf_constants.legs
        fwdkin_ds.attrs["keypoint_names_per_leg"] = (
            nmf_constants.leg_keypoints_canonical
        )

        if frame_ids is not None:
            f.create_dataset(
                "frame_ids",
                data=frame_ids,
                dtype=np.int32,
                compression="gzip",
                shuffle=True,
            )
