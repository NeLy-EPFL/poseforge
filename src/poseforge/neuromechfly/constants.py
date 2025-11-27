import numpy as np


###########################################################################
##  NEUROMECHFLY BODY CONFIGURATION BELOW                                ##
###########################################################################

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

legs = [f"{side}{pos}" for side in "LR" for pos in "FMH"]
leg_keypoints_canonical = ["ThC", "CTr", "FTi", "TiTa", "Claw"]
leg_keypoints_nmf = [
    keypoint_name_lookup_canonical_to_nmf[link] for link in leg_keypoints_canonical
]
keypoint_segments_canonical = [
    f"{leg}{link}" for leg in legs for link in leg_keypoints_canonical
] + ["LPedicel", "RPedicel"]
keypoint_segments_nmf = [
    f"{leg}{link}" for leg in legs for link in leg_keypoints_nmf
] + ["LPedicel", "RPedicel"]

all_segment_names_per_leg = [
    "Coxa",
    "Femur",
    "Tibia",
    "Tarsus1",
    "Tarsus2",
    "Tarsus3",
    "Tarsus4",
    "Tarsus5",
]

all_leg_dofs = [
    f"joint_{side}{pos}{dof}"
    for side in "LR"
    for pos in "FMH"
    for dof in [
        "Coxa",
        "Coxa_roll",
        "Coxa_yaw",
        "Femur",
        "Femur_roll",
        "Tibia",
        "Tarsus1",
    ]
]

kchain_plotting_colors = {  # these are only for plotting aesthetics
    "LF": np.array([15, 115, 153]) / 255,
    "LM": np.array([26, 141, 175]) / 255,
    "LH": np.array([117, 190, 203]) / 255,
    "RF": np.array([186, 30, 49]) / 255,
    "RM": np.array([201, 86, 79]) / 255,
    "RH": np.array([213, 133, 121]) / 255,
    "LAntenna": np.array([50, 120, 32]) / 255,
    "RAntenna": np.array([50, 120, 32]) / 255,
}


###########################################################################
##  COLORS FOR BODY SEGMENT RENDERING BELOW                              ##
##  These are set to artificially boost contrast between body segments   ##
##  -- they are NOT just for aesthetics!                                 ##
###########################################################################

# Define color combo by body segment
color_by_link = {
    "Coxa": "cyan",
    "Femur": "yellow",
    "Tibia": "blue",
    "Tarsus": "green",
    "Antenna": "magenta",
    "Thorax": "gray",
}
color_by_kinematic_chain = {
    "LF": "red",  # left front leg
    "LM": "green",  # left mid leg
    "LH": "blue",  # left hind leg
    "RF": "cyan",  # right front leg
    "RM": "magenta",  # right mid leg
    "RH": "yellow",  # right hind leg
    "L": "red",  # left antenna
    "R": "green",  # right antenna
    "Thorax": "white",  # thorax
}
color_palette = {
    "red": (1, 0, 0, 1),
    "green": (0, 1, 0, 1),
    "blue": (0, 0, 1, 1),
    "yellow": (1, 1, 0, 1),
    "magenta": (1, 0, 1, 1),
    "cyan": (0, 1, 1, 1),
    "gray": (0.4, 0.4, 0.4, 1),
    "white": (1, 1, 1, 1),
}


###########################################################################
##  PARAMETERS FOR INVERSE KINEMATICS WITH SEQIKPY BELOW                 ##
###########################################################################

# SeqIKPy considers the anchor point of every DoF a "joint" keypoint. However, some
# anatomical joints have multiple DoFs (e.g., ThC has yaw, pitch, roll). This results in
# some "virtual" keypoints in the inverse kinematics output. This mask filters them out.
# The keypoints in seqikpy output (including virtual ones) are:
#   0. ThC base (physical)
#   1. ThC pitch (virtual)
#   2. ThC roll (virtual)
#   3. ThC yaw (virtual)
#   4. CTr pitch (physical)
#   5. CTr roll (virtual)
#   6. FTi pitch (physical)
#   7. TiTa pitch (physical)
#   8. Claw (physical)
physical_keypoints_mask = np.array([1, 0, 0, 0, 1, 0, 1, 1, 1], dtype=bool)


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


# Source of hardcoded values below are taken from NeuroMechFly v2 (by Alfie)
# https://github.com/NeLy-EPFL/nmf2-paper/blob/961a64eb579d0dbb992de145771e33a698259e4a/revision_stepping/adapt_ik_to_locomotion_flytracker.ipynb
# fmt: off
nmf_initial_angles = {
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

# Define a template to create the kinematic chain for SeqIKPy

# The length of chain comes from the size calculated from the template
nmf_template = {
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
nmf_bounds = {
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

nmf_size = {
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
