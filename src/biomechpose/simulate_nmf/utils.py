import numpy as np


dof_name_lookup_nmf_to_canonical = {
    "Coxa": "ThC_pitch",
    "Coxa_roll": "ThC_roll",
    "Coxa_yaw": "ThC_yaw",
    "Femur": "CTr_pitch",
    "Femur_roll": "CTr_roll",
    "Tibia": "FTi_pitch",
    "Tarsus1": "TiTa_pitch",
}
dof_name_lookup_canonical_to_nmf = {
    v: k for k, v in dof_name_lookup_nmf_to_canonical.items()
}

keypoint_name_lookup_nmf_to_canonical = {
    "Coxa": "ThC",
    "Femur": "CTr",
    "Tibia": "FTi",
    "Tarsus1": "TiTa",
    "Tarsus5": "Claw",
    "Pedicel": "HeadAnt",
}
keypoint_name_lookup_canonical_to_nmf = {
    v: k for k, v in keypoint_name_lookup_nmf_to_canonical.items()
}

kinematic_chain_colors = {
    "LF": np.array([15, 115, 153]) / 255,
    "LM": np.array([26, 141, 175]) / 255,
    "LH": np.array([117, 190, 203]) / 255,
    "RF": np.array([186, 30, 49]) / 255,
    "RM": np.array([201, 86, 79]) / 255,
    "RH": np.array([213, 133, 121]) / 255,
    "LAntenna": np.array([255, 255, 0]) / 255,
    "RAntenna": np.array([255, 255, 0]) / 255,
}


def parse_nmf_joint_name(nmf_joint_name: str) -> tuple[str, str]:
    """Parse a NeuromechFly joint name (e.g. "joint_LFCoxa" or "LFCoxa")
    into a tuple of leg and canonical DoF name (e.g. ("LF", "ThC_pitch")).
    The latter is also used in Aymanns et al. 2022."""
    nmf_dof_name = nmf_joint_name.replace("joint_", "")
    leg = nmf_dof_name[:2]
    nmf_dof_name_no_leg = nmf_dof_name[2:]
    canonical_dof_name = dof_name_lookup_nmf_to_canonical[nmf_dof_name_no_leg]
    return leg, canonical_dof_name


def parse_nmf_keypoint_name(nmf_keypoint_name: str) -> tuple[str | None, str]:
    """Parse a NeuromechFly keypoint name (e.g. "LFCoxa" or "LFTarsus5")
    into canonical keypoint name (e.g. ("LF", "ThC") or ("LF", "Claw"))."""
    if nmf_keypoint_name[:3].isupper():  # leg: e.g. LFCoxa
        kchain = nmf_keypoint_name[:2]
        nmf_keypoint_name_no_leg = nmf_keypoint_name[2:]
        link = keypoint_name_lookup_nmf_to_canonical[nmf_keypoint_name_no_leg]
    elif nmf_keypoint_name[:2].isupper():  # single side, e.g. LEye
        kchain = nmf_keypoint_name[0]
        link = keypoint_name_lookup_nmf_to_canonical[nmf_keypoint_name[1:]]
    elif nmf_keypoint_name[0].isupper():  # no side, e.g. Thorax
        kchain = None
        link = keypoint_name_lookup_nmf_to_canonical[nmf_keypoint_name]
    else:
        raise ValueError(
            f"Cannot parse NeuroMechFly keypoint name: {nmf_keypoint_name}"
        )
    return kchain, link
