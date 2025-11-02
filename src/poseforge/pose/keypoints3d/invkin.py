import numpy as np
import h5py
import logging
from collections import defaultdict
from pathlib import Path
from seqikpy.alignment import AlignPose
from seqikpy.kinematic_chain import KinematicChainSeq
from seqikpy.leg_inverse_kinematics import LegInvKinSeq

from poseforge.neuromechfly.constants import (
    legs,
    leg_keypoints_nmf,
    leg_keypoints_canonical,
    dof_name_lookup_canonical_to_nmf,
    keypoint_name_lookup_nmf_to_canonical,
)
from poseforge.pose.keypoints3d.visualizer import visualize_leg_segment_lengths


# Source of hardcoded values below are taken from NeuroMechFly v2 (by Alfie)
# https://github.com/NeLy-EPFL/nmf2-paper/blob/961a64eb579d0dbb992de145771e33a698259e4a/revision_stepping/adapt_ik_to_locomotion_flytracker.ipynb
# fmt: off
_nmf_initial_angles = {
    "RF": {
        # Base ThC yaw pitch CTr pitch
        "stage_1": np.array([0.0, 0.45, -0.07, -2.14]),
        # Base ThC yaw pitch roll CTr pitch CTr roll
        "stage_2": np.array([0.0, 0.45, -0.07, -0.32, -2.14, 1.4]),
        # Base ThC yaw pitch roll CTr pitch CTr roll FTi pitch
        "stage_3": np.array([0.0, 0.45, -0.07, -0.32, -2.14, -1.25, 1.48, 0.0]),
        # Base ThC yaw pitch roll CTr pitch CTr roll FTi pitch TiTa pitch
        "stage_4": np.array([0.0, 0.45, -0.07, -0.32, -2.14, -1.25, 1.48, 0.0, 0.0]),
    },
    "LF": {
        "stage_1": np.array([0.0, -0.45, -0.07, -2.14]),
        "stage_2": np.array([0.0, -0.45, -0.07, 0.32, -2.14, 1.4]),
        "stage_3": np.array([0.0, -0.45, -0.07, 0.32, -2.14, 1.25, 1.48, 0.0]),
        "stage_4": np.array([0.0, -0.45, -0.07, 0.32, -2.14, 1.25, 1.48, 0.0, 0.0]),
    },
    "RM": {
        "stage_1": np.array([0.0, 0.45, 0.37, -2.14]),
        "stage_2": np.array([0.0, 0.45, 0.37, -0.32, -2.14, 1.4]),
        "stage_3": np.array([0.0, 0.45, 0.37, -0.32, -2.14, -1.25, 1.48, 0.0]),
        "stage_4": np.array([0.0, 0.45, 0.37, -0.32, -2.14, -1.25, 1.48, 0.0, 0.0]),
    },
    "LM": {
        "stage_1": np.array([0.0, -0.45, 0.37, -2.14]),
        "stage_2": np.array([0.0, -0.45, 0.37, 0.32, -2.14, 1.4]),
        "stage_3": np.array([0.0, -0.45, 0.37, 0.32, -2.14, 1.25, 1.48, 0.0]),
        "stage_4": np.array([0.0, -0.45, 0.37, 0.32, -2.14, 1.25, 1.48, 0.0, 0.0]),
    },
    "RH": {
        "stage_1": np.array([0.0, 0.45, 0.07, -2.14]),
        "stage_2": np.array([0.0, 0.45, 0.07, -0.32, -2.14, 1.4]),
        "stage_3": np.array([0.0, 0.45, 0.07, -0.32, -2.14, -1.25, 1.48, 0.0]),
        "stage_4": np.array([0.0, 0.45, 0.07, -0.32, -2.14, -1.25, 1.48, 0.0, 0.0]),
    },
    "LH": {
        "stage_1": np.array([0.0, -0.45, 0.07, -2.14]),
        "stage_2": np.array([0.0, -0.45, 0.07, 0.32, -2.14, 1.4]),
        "stage_3": np.array([0.0, -0.45, 0.07, 0.32, -2.14, 1.25, 1.48, 0.0]),
        "stage_4": np.array([0.0, -0.45, 0.07, 0.32, -2.14, 1.25, 1.48, 0.0, 0.0]),
    },
    "head": np.array([0, -0.17, 0]),  #  none, roll, pitch, yaw
}

# Define a template to create the kinematic chain
# The length of chain comes from the size calculated from the template

_nmf_template = {
    "RF_Coxa": np.array([0.35, -0.27, 0.400]),
    "RF_Femur": np.array([0.35, -0.27, -0.025]),
    "RF_Tibia": np.array([0.35, -0.27, -0.731]),
    "RF_Tarsus": np.array([0.35, -0.27, -1.249]),
    "RF_Claw": np.array([0.35, -0.27, -1.912]),
    "LF_Coxa": np.array([0.35, 0.27, 0.400]),
    "LF_Femur": np.array([0.35, 0.27, -0.025]),
    "LF_Tibia": np.array([0.35, 0.27, -0.731]),
    "LF_Tarsus": np.array([0.35, 0.27, -1.249]),
    "LF_Claw": np.array([0.35, 0.27, -1.912]),
    "RM_Coxa": np.array([0, -0.125, 0]),
    "RM_Femur": np.array([0, -0.125, -0.182]),
    "RM_Tibia": np.array([0, -0.125, -0.965]),
    "RM_Tarsus": np.array([0, -0.125, -1.633]),
    "RM_Claw": np.array([0, -0.125, -2.328]),
    "LM_Coxa": np.array([0, 0.125, 0]),
    "LM_Femur": np.array([0, 0.125, -0.182]),
    "LM_Tibia": np.array([0, 0.125, -0.965]),
    "LM_Tarsus": np.array([0, 0.125, -1.633]),
    "LM_Claw": np.array([0, 0.125, -2.328]),
    "RH_Coxa": np.array([-0.215, -0.087, -0.073]),
    "RH_Femur": np.array([-0.215, -0.087, -0.272]),
    "RH_Tibia": np.array([-0.215, -0.087, -1.108]),
    "RH_Tarsus": np.array([-0.215, -0.087, -1.793]),
    "RH_Claw": np.array([-0.215, -0.087, -2.588]),
    "LH_Coxa": np.array([-0.215, 0.087, -0.073]),
    "LH_Femur": np.array([-0.215, 0.087, -0.272]),
    "LH_Tibia": np.array([-0.215, 0.087, -1.108]),
    "LH_Tarsus": np.array([-0.215, 0.087, -1.793]),
    "LH_Claw": np.array([-0.215, 0.087, -2.588]),
}

# Determine the bounds for each joint DOF
_nmf_bounds = {
    # Front legs
    "RF_ThC_yaw": (np.deg2rad(-45), np.deg2rad(45)),
    "RF_ThC_pitch": (np.deg2rad(-10), np.deg2rad(90)),
    "RF_ThC_roll": (np.deg2rad(-135), np.deg2rad(10)),  # ? 1
    "RF_CTr_pitch": (np.deg2rad(-270), np.deg2rad(10)),  # ? 2
    "RF_CTr_roll": (np.deg2rad(-180), np.deg2rad(90)),  # ? 3
    "RF_FTi_pitch": (np.deg2rad(-10), np.deg2rad(180)),
    "RF_TiTa_pitch": (np.deg2rad(-180), np.deg2rad(10)),
    "LF_ThC_yaw": (np.deg2rad(-45), np.deg2rad(45)),
    "LF_ThC_pitch": (np.deg2rad(-10), np.deg2rad(90)),
    "LF_ThC_roll": (np.deg2rad(-10), np.deg2rad(90)),  # ? 1
    "LF_CTr_pitch": (np.deg2rad(-180), np.deg2rad(10)),  # ? 2
    "LF_CTr_roll": (np.deg2rad(-90), np.deg2rad(180)),  # ? 3
    "LF_FTi_pitch": (np.deg2rad(-10), np.deg2rad(180)),
    "LF_TiTa_pitch": (np.deg2rad(-180), np.deg2rad(10)),
    
    # Mid legs
    "RM_ThC_yaw": (np.deg2rad(-45), np.deg2rad(45)),  # ? 4
    "RM_ThC_pitch": (np.deg2rad(-10), np.deg2rad(90)),
    "RM_ThC_roll": (np.deg2rad(-180), np.deg2rad(10)),  # ? 5
    "RM_CTr_pitch": (np.deg2rad(-270), np.deg2rad(10)),  # ? 6
    "RM_CTr_roll": (np.deg2rad(-90), np.deg2rad(90)),
    "RM_FTi_pitch": (np.deg2rad(-10), np.deg2rad(180)),
    "RM_TiTa_pitch": (np.deg2rad(-180), np.deg2rad(10)),
    "LM_ThC_yaw": (np.deg2rad(-45), np.deg2rad(90)),  # ? 4
    "LM_ThC_pitch": (np.deg2rad(-10), np.deg2rad(90)),
    "LM_ThC_roll": (np.deg2rad(-10), np.deg2rad(180)),  # ? 5
    "LM_CTr_pitch": (np.deg2rad(-180), np.deg2rad(10)),  # ? 6
    "LM_CTr_roll": (np.deg2rad(-90), np.deg2rad(90)),
    "LM_FTi_pitch": (np.deg2rad(-10), np.deg2rad(180)),
    "LM_TiTa_pitch": (np.deg2rad(-180), np.deg2rad(10)),
    
    # Hind legs
    "RH_ThC_yaw": (np.deg2rad(-45), np.deg2rad(45)),  # ? 7
    "RH_ThC_pitch": (np.deg2rad(-10), np.deg2rad(90)),
    "RH_ThC_roll": (np.deg2rad(-180), np.deg2rad(10)),  # ? 8
    "RH_CTr_pitch": (np.deg2rad(-180), np.deg2rad(10)),
    "RH_CTr_roll": (np.deg2rad(-90), np.deg2rad(90)),
    "RH_FTi_pitch": (np.deg2rad(-10), np.deg2rad(180)),
    "RH_TiTa_pitch": (np.deg2rad(-180), np.deg2rad(10)),
    "LH_ThC_yaw": (np.deg2rad(-45), np.deg2rad(90)),  # ? 7
    "LH_ThC_pitch": (np.deg2rad(-10), np.deg2rad(90)),
    "LH_ThC_roll": (np.deg2rad(-10), np.deg2rad(180)),  # ? 8
    "LH_CTr_pitch": (np.deg2rad(-180), np.deg2rad(10)),
    "LH_CTr_roll": (np.deg2rad(-90), np.deg2rad(90)),
    "LH_FTi_pitch": (np.deg2rad(-10), np.deg2rad(180)),
    "LH_TiTa_pitch": (np.deg2rad(-180), np.deg2rad(10)),
}

_nmf_size = {
    "RF_Coxa": 0.40, "RM_Coxa": 0.182, "RH_Coxa": 0.199,
    "LF_Coxa": 0.40, "LM_Coxa": 0.182, "LH_Coxa": 0.199,
    "RF_Femur": 0.69, "RM_Femur": 0.783, "RH_Femur": 0.836,
    "LF_Femur": 0.69, "LM_Femur": 0.783, "LH_Femur": 0.836,
    "RF_Tibia": 0.54, "RM_Tibia": 0.668, "RH_Tibia": 0.685,
    "LF_Tibia": 0.54, "LM_Tibia": 0.668, "LH_Tibia": 0.685,
    "RF_Tarsus": 0.63, "RM_Tarsus": 0.695, "RH_Tarsus": 0.795,
    "LF_Tarsus": 0.63, "LM_Tarsus": 0.695, "LH_Tarsus": 0.795,
    "RF": 2.26, "RM": 2.328, "RH": 2.515,
    "LF": 2.26, "LM": 2.328, "LH": 2.515,
    "Antenna": 0.2745906043549196, "Antenna_mid_thorax": 0.9355746896961248,
}
# fmt: on


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
    for leg in legs:
        data_block = np.full(
            (n_frames, len(leg_keypoints_nmf), 3), np.nan, dtype=np.float32
        )
        for keypoint_idx, keypoint_name in enumerate(leg_keypoints_nmf):
            poseforge_key = (
                f"{leg}{keypoint_name_lookup_nmf_to_canonical[keypoint_name]}"
            )
            idx = keypoint_names_canonical.index(poseforge_key)
            data_block[:, keypoint_idx, :] = world_xyz[:n_frames, idx, :]
        assert not np.isnan(data_block).any()
        pose_data_dict[f"{leg}_leg"] = data_block

    return pose_data_dict


def extract_leg_segment_lengths(pose_data_dict: dict[str, np.ndarray]):
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


def scale_template_by_leg_segment_sizes(template_positions_dict, size_dict):
    logger = logging.getLogger(__name__)

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
                pos_new = prev_pos_original + scaled_vector
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
) -> dict[str, np.ndarray]:
    # Convert input format to what SeqIKPy expects
    pose_data_dict = _world_xyz_to_seqikpy_format(
        world_xyz, keypoint_names_canonical, max_n_frames=max_n_frames
    )

    leg_segment_lengths_over_time = extract_leg_segment_lengths(pose_data_dict)
    sizes_from_data = calculate_average_leg_segment_lengths(
        leg_segment_lengths_over_time, make_symmetric=True
    )
    for k, v in _nmf_size.items():
        if k not in sizes_from_data:
            sizes_from_data[k] = v  # add antennae sizes
    template_scaled_to_data = scale_template_by_leg_segment_sizes(
        _nmf_template, sizes_from_data
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
        legs_list=legs,
        include_claw=True,
        body_template=template_scaled_to_data,
        body_size=sizes_from_data,
    )
    aligned_pose = pose_aligner.align_pose()

    # Run inverse kinematics and forward kinematics.
    # This is the slowest part and it's parallelized at leg level (so use n_workers=6).
    seq_kinematic_chain = KinematicChainSeq(
        bounds_dof=_nmf_bounds, legs_list=legs, body_size=sizes_from_data
    )
    leg_seq_ik = LegInvKinSeq(
        aligned_pos=aligned_pose,
        kinematic_chain_class=seq_kinematic_chain,
        initial_angles=_nmf_initial_angles,
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
    dof_names_per_leg = list(dof_name_lookup_canonical_to_nmf.keys())
    n_frames_joint_angles = joint_angles["Angle_LF_ThC_yaw"].shape[0]
    joint_angles_arr = np.full((n_frames_joint_angles, 6, 7), np.nan, dtype=np.float32)
    for leg_idx, leg in enumerate(legs):
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
    n_physical_keypoints = len(leg_keypoints_nmf)
    n_frames_fwdkin, n_virtual_keypoints, _ = forward_kinematics["LF_leg"].shape
    assert n_frames_fwdkin == n_frames_joint_angles
    assert n_physical_keypoints == 5
    assert n_virtual_keypoints == 9
    fwdkin_world_xyz = np.full(
        (n_frames_fwdkin, len(legs), n_virtual_keypoints, 3), np.nan, dtype=np.float32
    )
    for leg_idx, leg in enumerate(legs):
        fwdkin_world_xyz[:, leg_idx, :, :] = forward_kinematics[f"{leg}_leg"]
    assert not np.isnan(fwdkin_world_xyz).any()
    # Drop virtual keypoints that share the same physical location
    is_physical = np.array([1, 0, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    fwdkin_world_xyz = fwdkin_world_xyz[:, :, is_physical, :]

    with h5py.File(output_path, "w") as f:
        joint_angles_ds = f.create_dataset(
            "joint_angles", data=joint_angles_arr, dtype=np.float32, compression="gzip"
        )
        joint_angles_ds.attrs["legs"] = legs
        joint_angles_ds.attrs["dof_names_per_leg"] = dof_names_per_leg

        fwdkin_ds = f.create_dataset(
            "forward_kinematics_world_xyz",
            data=fwdkin_world_xyz,
            dtype=np.float32,
            compression="gzip",
        )
        fwdkin_ds.attrs["legs"] = legs
        fwdkin_ds.attrs["keypoint_names_per_leg"] = leg_keypoints_canonical

        if frame_ids is not None:
            f.create_dataset(
                "frame_ids",
                data=frame_ids,
                dtype=np.int32,
                compression="gzip",
                shuffle=True,
            )
