import numpy as np


dof_name_lookup_neuromechfly_to_canonical = {
    "Coxa": "ThC_pitch",
    "Coxa_roll": "ThC_roll",
    "Coxa_yaw": "ThC_yaw",
    "Femur": "CTr_pitch",
    "Femur_roll": "CTr_roll",
    "Tibia": "FTi_pitch",
    "Tarsus1": "TiTa_pitch",
}

keypoint_name_lookup_neuromechfly_to_canonical = {
    "Coxa": "ThC",
    "Femur": "CTr",
    "Tibia": "FTi",
    "Tarsus1": "TiTa",
    "Tarsus5": "Claw",
}

leg_colors = {
    "LF": np.array([15, 115, 153]) / 255,
    "LM": np.array([26, 141, 175]) / 255,
    "LH": np.array([117, 190, 203]) / 255,
    "RF": np.array([186, 30, 49]) / 255,
    "RM": np.array([201, 86, 79]) / 255,
    "RH": np.array([213, 133, 121]) / 255,
}


def parse_nmf_joint_name(nmf_joint_name: str) -> tuple[str, str]:
    """Parse a NeuromechFly joint name (e.g. "joint_LFCoxa" or "LFCoxa")
    into a tuple of leg and canonical DoF name (e.g. ("LF", "ThC_pitch")).
    The latter is also used in Aymanns et al. 2022."""
    nmf_dof_name = nmf_joint_name.replace("joint_", "")
    leg = nmf_dof_name[:2]
    nmf_dof_name_no_leg = nmf_dof_name[2:]
    canonical_dof_name = dof_name_lookup_neuromechfly_to_canonical[nmf_dof_name_no_leg]
    return leg, canonical_dof_name


def parse_nmf_keypoint_name(nmf_keypoint_name: str) -> tuple[str, str]:
    """Parse a NeuromechFly keypoint name (e.g. "LFCoxa" or "LFTarsus5")
    into canonical keypoint name (e.g. ("LF", "ThC") or ("LF", "Claw"))."""
    leg = nmf_keypoint_name[:2]
    nmf_keypoint_name_no_leg = nmf_keypoint_name[2:]
    canonical_keypoint_name = keypoint_name_lookup_neuromechfly_to_canonical[
        nmf_keypoint_name_no_leg
    ]
    return leg, canonical_keypoint_name
