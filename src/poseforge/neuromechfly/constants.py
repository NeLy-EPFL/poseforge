import numpy as np
from flygym.anatomy import JointDOF


###########################################################################
##  NEUROMECHFLY BODY CONFIGURATION BELOW                                ##
###########################################################################

keypoint_name_lookup_nmf_to_canonical = {
    "coxa": "ThC",
    "trochanterfemur": "CTr",
    "tibia": "FTi",
    "tarsus1": "TiTa",
    "tarsus5": "Claw",
    "pedicel": "HeadAnt",
}

keypoint_name_lookup_canonical_to_nmf = {
    v: k for k, v in keypoint_name_lookup_nmf_to_canonical.items()
    }

# Mapping from the canonical (Aymanns et al. 2022) leg DOF names to the
# NeuroMechFly DOF (joint) names.
#
# This constant was originally introduced in commit 5f8b6ee and accidentally
# removed in 3c9449a ("compatibility with flygym v2") while its call sites in
# `run_inverse_kinematics.py` and `production/spotlight/keypoints3d.py` were
# never updated -> they raised AttributeError when saving IK output. Restored
# here (see issue #48, finding I3-A).
#
# IMPORTANT - ordering and key names:
#   * The KEYS are the 7 canonical leg DOF names. They are kept in the order
#       ThC_yaw, ThC_pitch, ThC_roll, CTr_pitch, CTr_roll, FTi_pitch, TiTa_pitch
#     because the call sites do
#       `dof_names_per_leg = list(dof_name_lookup_canonical_to_nmf.keys())`
#     and use that order as the DOF axis of the saved `joint_angles` array
#     (and as its `dof_names_per_leg` attribute). This is the same DOF ordering
#     used by `nmf_initial_angles` / seqikpy's kinematic chain (yaw, pitch,
#     roll, ...).
#   * seqikpy emits joint-angle dict keys of the form
#       `Angle_{leg}_{canonical_dof}` (e.g. "Angle_LF_ThC_yaw"); see
#       seqikpy.leg_inverse_kinematics.LegInvKinSeq. The call sites build the
#       lookup key as `f"Angle_{leg}_{dof_name}"` where `dof_name` is a KEY of
#       this dict, so the keys must be exactly these canonical DOF names for the
#       lookups (and hence the DOF packing order) to be correct.
#   * The VALUES are the corresponding NeuroMechFly DOF names. They are not used
#     by the IK save path today but document the canonical<->NMF correspondence
#     and keep this dict useful for downstream NMF actuation code.
dof_name_lookup_canonical_to_nmf = {
    "ThC_yaw": "Coxa_yaw",
    "ThC_pitch": "Coxa",
    "ThC_roll": "Coxa_roll",
    "CTr_pitch": "Femur",
    "CTr_roll": "Femur_roll",
    "FTi_pitch": "Tibia",
    "TiTa_pitch": "Tarsus1",
}
dof_name_lookup_nmf_to_canonical = {
    v: k for k, v in dof_name_lookup_canonical_to_nmf.items()
}

legs = [f"{side}{pos}" for side in "LR" for pos in "FMH"]
leg_keypoints_canonical = ["ThC", "CTr", "FTi", "TiTa", "Claw"]
leg_keypoints_nmf = [keypoint_name_lookup_canonical_to_nmf[kp] for kp in leg_keypoints_canonical]
keypoint_segments_canonical = [
    f"{leg}{link.replace("-", "")}" for leg in legs for link in leg_keypoints_canonical
] + ["LPedicel", "RPedicel"]
keypoint_segments_nmf = [
    f"{leg.lower()}{link}" for leg in legs for link in leg_keypoints_nmf
] + ["LPedicel", "RPedicel"]

keypoint_segments_nmf = [f"{leg.lower()}_{kp}" for leg in legs for kp in leg_keypoints_nmf] + ["l_pedicel", "r_pedicel"]

keypoint_segments_flybody = [f"{leg.lower()}_{kp}" for leg in legs for kp in leg_keypoints_nmf] + ["l_antenna", "r_antenna"]

# all_segment_names_per_leg = [
#     "Coxa",
#     "Femur",
#     "Tibia",
#     "Tarsus1",
#     "Tarsus2",
#     "Tarsus3",
#     "Tarsus4",
#     "Tarsus5",
# ]

# all_leg_dofs = [
#     f"joint_{side}{pos}{dof}"
#     for side in "LR"
#     for pos in "FMH"
#     for dof in [
#         "Coxa",
#         "Coxa_roll",
#         "Coxa_yaw",
#         "Femur",
#         "Femur_roll",
#         "Tibia",
#         "Tarsus1",
#     ]
# ]

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
    "coxa": "cyan",
    "trochanterfemur": "yellow",
    "tibia": "blue",
    "tarsus": "green",
    "antenna": "magenta",
    "thorax": "gray",
}
color_by_kinematic_chain = {
    "lf_": "red",  # left front leg
    "lm_": "green",  # left mid leg
    "lh_": "blue",  # left hind leg
    "rf_": "cyan",  # right front leg
    "rm_": "magenta",  # right mid leg
    "rh_": "yellow",  # right hind leg
    "l_": "red",  # left antenna
    "r_": "green",  # right antenna
    "thorax": "white",  # thorax
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

joint_segments_nmf_to_canonical = {
    "thorax":"Th",
    "coxa":"C",
    "trochanterfemur_parent":"F",
    "trochanterfemur_child":"Tr",
    "tibia":"Ti",
    "tarsus1":"Ta",
    "pedicel":"HeadAnt",
    "tarsus5":"Claw",
}
def parse_nmf_joint_seg(parent_name:str, child_name:str, dof:str) -> tuple[str, str]:
    """Parse a NeuroMechFly joint name (e.g. parent: "c_thorax", child: "lf_coxa", dof: "yaw") into
    Aymanns et al. 2022 joint name, return leg and the DOF name.

    Args:
        parent_name: e.g. "c_thorax"
        child_name: e.g. "lf_coxa"
        dof: e.g. "yaw"

    Returns:
        leg: e.g. "LF"
        aymanns_dof: e.g. "ThC_yaw"
    """
    _, parent_seg = parent_name.split("_")
    leg, child_seg = child_name.split("_")
    if child_seg=="trochanterfemur":
        child_seg = "trochanterfemur_child"
    if parent_seg=="trochanterfemur":
        parent_seg = "trochanterfemur_parent"
    leg = leg.upper()  # e.g. "lf" -> "LF"
    p_canonical = joint_segments_nmf_to_canonical[parent_seg]
    c_canonical = joint_segments_nmf_to_canonical[child_seg]
    dof_low = dof.lower()
    aymanns_dof = f"{p_canonical}{c_canonical}_{dof_low}"  # e.g. "ThC_CTr_yaw"
    return leg, aymanns_dof


def parse_nmf_joint(joint: JointDOF) -> tuple[str, str]:
    """Parse a joint to extract Aymanns et al. 2022 joint name, return
    leg and the DOF name.

    Args:
        joint: joint from skeleton.get_actuated_dofs_from_preset() with .name attribute
        e.g. "c_thorax-lf_coxa-yaw"

    Returns:
        leg: e.g. "LF"
        aymanns_dof: e.g. "ThC_yaw"
    """

    parent, child = joint.parent.name, joint.child.name
    return parse_nmf_joint_seg(parent, child, joint.axis.name)


# def parse_nmf_keypoint_name(nmf_keypoint_name: str) -> tuple[str | None, str]:
#     """Parse a NeuromechFly keypoint name (e.g. "lf_coxa" or "lf_tarsus5")
#     into canonical keypoint name (e.g. ("lf", "coxa") or ("lf", "tarsus5"))."""
#     if nmf_keypoint_name[:3].isupper():  # leg: e.g. LFCoxa
#         kchain = nmf_keypoint_name[:2]
#         nmf_keypoint_name_no_leg = nmf_keypoint_name[2:]
#         link = keypoint_name_lookup_nmf_to_canonical[nmf_keypoint_name_no_leg]
#     elif nmf_keypoint_name[:2].isupper():  # single side, e.g. LEye
#         kchain = nmf_keypoint_name[0]
#         link = keypoint_name_lookup_nmf_to_canonical[nmf_keypoint_name[1:]]
#     elif nmf_keypoint_name[0].isupper():  # no side, e.g. Thorax
#         kchain = None
#         link = keypoint_name_lookup_nmf_to_canonical[nmf_keypoint_name]
#     else:
#         raise ValueError(
#             f"Cannot parse NeuroMechFly keypoint name: {nmf_keypoint_name}"
#         )
#     return kchain, link


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

# Joint DOF bounds for seqikpy IK. DATA-DERIVED (issue #48 I3-B).
#
# Method (values are whole degrees wrapped in np.deg2rad):
#   1. Source: the ground-truth simulated `dof_angles` stored in every atomic-batch
#      `_labels.h5` (named via the dataset's `keys` attr; 6 legs x 7 DOFs = 42).
#   2. Sample: 500 `_labels.h5` files (numpy default_rng(0), no replacement) from the
#      sorted recursive glob of bulk_data/.../atomic_batches/4variants/**/*_labels.h5,
#      all 32 frames each -> n = 16000 frames.
#   3. Per DOF: bound = (floor(min_deg - margin), ceil(max_deg + margin)) with
#      margin = 10 deg -- the observed range padded outward. The margin gives the
#      least-squares solver headroom so the optimum does not sit exactly on a boundary
#      (an IK fragility noted in the audit); min/max (not percentiles) guarantee no
#      observed pose is clipped.
#   4. Map ground-truth key `{leg}{dof}` (e.g. RFThC_yaw) -> bounds key `{leg}_{dof}`.
#   Reproduce EXACTLY with:
#     python scripts/verify_ik_selfconsistency.py --emit-bounds --n-batches 500 --seed 0 --margin-deg 10
#
# Why: supersedes the earlier hand-set / L-R-mirrored bounds, 10/42 of which were tighter
# than the actual range of motion and would clip valid poses (a bound tighter than the data
# is unreachable by IK -> forces a wrong solution). The simulated dof_angles are exactly the
# RoM the IK must reproduce, so they are the correct floor for these bounds.
# CAVEATS: (1) this is the *training* RoM, not the anatomical RoM; production may see novel
#   poses, so widen toward NeuroMechFly's anatomical limits if IK saturates a bound.
#   (2) DOFs flagged WRAPPING below have source angles beyond +/-180 deg, indicating the
#   upstream kinematics need unwrapping; bounds contain them only so IK can reproduce them.
#   Wrapping DOFs: RH_ThC_roll. (3) Bounds come out near-mirror L/R where the data is, but
#   are NOT forced symmetric (data-honest).
nmf_bounds = {
    # Front legs
    "RF_ThC_yaw": (np.deg2rad(-36), np.deg2rad(57)),
    "RF_ThC_pitch": (np.deg2rad(-17), np.deg2rad(79)),
    "RF_ThC_roll": (np.deg2rad(-185), np.deg2rad(105)),
    "RF_CTr_pitch": (np.deg2rad(-187), np.deg2rad(-41)),
    "RF_CTr_roll": (np.deg2rad(-187), np.deg2rad(13)),
    "RF_FTi_pitch": (np.deg2rad(5), np.deg2rad(178)),
    "RF_TiTa_pitch": (np.deg2rad(-147), np.deg2rad(9)),
    "LF_ThC_yaw": (np.deg2rad(-63), np.deg2rad(37)),
    "LF_ThC_pitch": (np.deg2rad(-21), np.deg2rad(65)),
    "LF_ThC_roll": (np.deg2rad(-18), np.deg2rad(172)),
    "LF_CTr_pitch": (np.deg2rad(-180), np.deg2rad(-51)),
    "LF_CTr_roll": (np.deg2rad(-21), np.deg2rad(186)),
    "LF_FTi_pitch": (np.deg2rad(-6), np.deg2rad(182)),
    "LF_TiTa_pitch": (np.deg2rad(-150), np.deg2rad(10)),
    # Mid legs
    "RM_ThC_yaw": (np.deg2rad(-45), np.deg2rad(28)),
    "RM_ThC_pitch": (np.deg2rad(-31), np.deg2rad(27)),
    "RM_ThC_roll": (np.deg2rad(-171), np.deg2rad(-29)),
    "RM_CTr_pitch": (np.deg2rad(-155), np.deg2rad(-50)),
    "RM_CTr_roll": (np.deg2rad(-78), np.deg2rad(12)),
    "RM_FTi_pitch": (np.deg2rad(0), np.deg2rad(166)),
    "RM_TiTa_pitch": (np.deg2rad(-81), np.deg2rad(9)),
    "LM_ThC_yaw": (np.deg2rad(-26), np.deg2rad(38)),
    "LM_ThC_pitch": (np.deg2rad(-32), np.deg2rad(26)),
    "LM_ThC_roll": (np.deg2rad(35), np.deg2rad(167)),
    "LM_CTr_pitch": (np.deg2rad(-157), np.deg2rad(-38)),
    "LM_CTr_roll": (np.deg2rad(-13), np.deg2rad(83)),
    "LM_FTi_pitch": (np.deg2rad(4), np.deg2rad(163)),
    "LM_TiTa_pitch": (np.deg2rad(-132), np.deg2rad(10)),
    # Hind legs
    "RH_ThC_yaw": (np.deg2rad(-65), np.deg2rad(39)),
    "RH_ThC_pitch": (np.deg2rad(-33), np.deg2rad(50)),
    "RH_ThC_roll": (np.deg2rad(-216), np.deg2rad(-62)),  # WRAPPING: source RoM exceeds +/-180 deg
    "RH_CTr_pitch": (np.deg2rad(-158), np.deg2rad(-27)),
    "RH_CTr_roll": (np.deg2rad(-16), np.deg2rad(167)),
    "RH_FTi_pitch": (np.deg2rad(-3), np.deg2rad(168)),
    "RH_TiTa_pitch": (np.deg2rad(-156), np.deg2rad(10)),
    "LH_ThC_yaw": (np.deg2rad(-28), np.deg2rad(66)),
    "LH_ThC_pitch": (np.deg2rad(-33), np.deg2rad(53)),
    "LH_ThC_roll": (np.deg2rad(63), np.deg2rad(187)),
    "LH_CTr_pitch": (np.deg2rad(-169), np.deg2rad(-10)),
    "LH_CTr_roll": (np.deg2rad(-115), np.deg2rad(85)),
    "LH_FTi_pitch": (np.deg2rad(2), np.deg2rad(168)),
    "LH_TiTa_pitch": (np.deg2rad(-123), np.deg2rad(9)),
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
