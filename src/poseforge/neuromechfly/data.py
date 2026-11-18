import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from poseforge.neuromechfly.constants import parse_nmf_joint #, all_leg_dofs
from flygym.anatomy import JointDOF

# Supported kinematic-recording input formats.
#
# "aymanns2022": the tethered-fly-on-a-ball recordings of Aymanns et al. 2022,
#     stored as a pickled pandas DataFrame whose joint-angle columns are named
#     "Angle__{leg}_leg_{aymanns_dof}" (e.g. "Angle__RF_leg_ThC_yaw"). Right-leg
#     roll and yaw angles use the opposite sign to the NeuroMechFly (flygym)
#     model and are negated on load.
# "deeperfly": the freely-moving single-fly hexaview recordings tracked with
#     deeperfly, stored as an HDF5 file whose "ik/angles" dataset already holds
#     joint angles defined directly on the NeuroMechFly (flygym) body model, in
#     radians, with flygym's own axes and zeros. No sign correction is needed,
#     and each angle column is named exactly like the corresponding flygym
#     JointDOF (e.g. "c_thorax-lf_coxa-yaw").
SOURCE_FORMAT_AYMANNS2022 = "aymanns2022"
SOURCE_FORMAT_DEEPERFLY = "deeperfly"
VALID_SOURCE_FORMATS = (SOURCE_FORMAT_AYMANNS2022, SOURCE_FORMAT_DEEPERFLY)

# The six legs, used to locate the thorax-coxa keypoints in deeperfly recordings.
DEEPERFLY_LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"]
# Acquisition artifact: the infrared illumination is dark for a moment right
# after the start of a deeperfly recording, so the first frames are dropped by
# convention (see the dataset README).
DEEPERFLY_DROP_INITIAL_FRAMES = 25
# A freely-moving fly is considered to be moving when its body speed exceeds
# this threshold. Below it, tracking jitter dominates (see the dataset README).
DEEPERFLY_MOVING_SPEED_THRESHOLD_MM_PER_SEC = 1.0
# Dorsal-direction z-component thresholds that classify which arena surface the
# fly is on: pointing up (floor), down (lid), or sideways (wall).
DEEPERFLY_SURFACE_DORSAL_Z_THRESHOLD = 0.7


def extract_joint_angles_trajectory(
    kinematic_recording_segment: pd.DataFrame,
    interp_factor: int,
    actuated_joints_list: list[JointDOF],
    use_flybody: bool = False,
    source_format: str = SOURCE_FORMAT_AYMANNS2022,
) -> np.ndarray:
    """Extract joint angles trajectory from a kinematic recording segment,
    interpolating the angles to match the simulation timestep.

    The returned trajectory is always expressed in the convention of the body
    model being simulated, so that it can be fed directly to the actuators. Any
    format-specific sign or naming conversion happens here, selected by
    ``source_format`` (see ``VALID_SOURCE_FORMATS``).
    """
    if source_format not in VALID_SOURCE_FORMATS:
        raise ValueError(
            f"Unknown source_format {source_format!r}; "
            f"expected one of {VALID_SOURCE_FORMATS}."
        )
    if source_format == SOURCE_FORMAT_DEEPERFLY and use_flybody:
        raise NotImplementedError(
            "deeperfly joint angles are defined on the NeuroMechFly (flygym) "
            "body model, so they cannot be used with the flybody model "
            "(use_flybody=True)."
        )

    # Extract the original joint angles from the kinematic recording segment
    original_trajectories_list = []
    n_joints = 0
    for joint in actuated_joints_list:
        if source_format == SOURCE_FORMAT_AYMANNS2022:
            leg, aymans_dof = parse_nmf_joint(joint)
            column_name = f"Angle__{leg}_leg_{aymans_dof}"
            time_series = kinematic_recording_segment[column_name].values
            if not use_flybody:
                if leg.startswith("R"):
                    if "roll" in aymans_dof:
                        time_series = -time_series  # Negate roll angles for right legs to match NMF convention
                    if "yaw" in aymans_dof:
                        time_series = -time_series  # Negate angles for right legs to match NMF convention
        else:  # SOURCE_FORMAT_DEEPERFLY
            # deeperfly angle columns are named exactly like the flygym JointDOF
            # and already use the flygym convention, so no sign flip is applied.
            column_name = joint.name
            if column_name not in kinematic_recording_segment.columns:
                raise KeyError(
                    f"Joint angle column {column_name!r} not found in the "
                    f"deeperfly segment. Available columns include "
                    f"{list(kinematic_recording_segment.columns)[:5]}..."
                )
            time_series = kinematic_recording_segment[column_name].values

        original_trajectories_list.append(time_series)
        n_joints += 1

    original_trajectories_array = np.array(original_trajectories_list).T
    # Interpolate the joint angles to match the simulation timestep
    num_frames = len(kinematic_recording_segment)
    interp_num_frames = num_frames * interp_factor
    interp_trajectories_array = np.zeros((interp_num_frames, n_joints))
    for i in range(original_trajectories_array.shape[1]):
        traj = original_trajectories_array[:, i]
        tarj_interp = np.interp(
            np.arange(interp_num_frames),
            np.arange(num_frames) * interp_factor,
            traj,
        )
        interp_trajectories_array[:, i] = tarj_interp

    return interp_trajectories_array


def load_kinematic_recording(
    recording_path: str | Path,
    min_duration_sec: float,
    input_timestep: float,
    filter_size: int = 5,
    filtered_frac_threshold: float = 0.5,
) -> list[pd.DataFrame]:
    """Load kinematic recording from a PKL file, split it into discrete,
    non-resting segments, and return a list of DataFrames, each one being
    the trajectory of a single segment.

    Args:
        recording_path: Path to the kinematic recording PKL file.
        min_duration_sec: Minimum duration (seconds) a segment must have
            to be included.
        input_timestep: Timestep of the input kinematic recording
            (seconds). E.g. 0.01 for 100 Hz recordings.
        filter_size: Size of the moving average filter to denoise the
            resting mask.
        filtered_frac_threshold: Threshold for the moving average filter to
            consider a segment as non-resting. If the mean of the filter
            exceeds this threshold, the segment is considered non-resting.

    Returns:
        A list of DataFrames, each containing a segment of non-resting
        kinematic trajectory (same format as PKL data).
    """
    kinematic_recording = pd.read_pickle(recording_path).reset_index()

    # Denoise the mask by convolving it with a moving average filter
    is_not_resting_mask = (kinematic_recording["Prediction"] != "resting").values
    is_not_resting_moving_average = np.convolve(
        is_not_resting_mask.astype(int),
        np.ones(filter_size) / filter_size,
        mode="same",
    )
    is_not_resting_mask_filtered = (
        is_not_resting_moving_average > filtered_frac_threshold
    )

    # Split the recording into non-resting segments based on the mask
    segments = []
    segment_start = None
    for i, is_not_resting in enumerate(is_not_resting_mask_filtered):
        if is_not_resting and segment_start is None:
            segment_start = i
        elif not is_not_resting and segment_start is not None:
            if i - segment_start >= min_duration_sec / input_timestep:
                segments.append(kinematic_recording.iloc[segment_start:i])
            segment_start = None

    return segments


def interpolate_trajectories(
    kinematic_recording_segment: pd.DataFrame,
    input_timestep: float,
    sim_timestep: float,
    actuated_joints_list: list[JointDOF],
    use_flybody: bool = False,
    source_format: str = SOURCE_FORMAT_AYMANNS2022,
):
    interp_factor = input_timestep / sim_timestep
    if int(interp_factor) != interp_factor:
        raise ValueError(
            f"Input timestep {input_timestep} must be a multiple of "
            f"simulation timestep {sim_timestep}."
        )
    interp_factor = int(interp_factor)
    trajectories_interp = extract_joint_angles_trajectory(
        kinematic_recording_segment,
        interp_factor,
        actuated_joints_list,
        use_flybody,
        source_format=source_format,
    )
    return trajectories_interp, interp_factor


def _split_active_mask_into_segments(
    is_active_mask: np.ndarray,
    min_frames: int,
    filter_size: int,
    filtered_frac_threshold: float,
) -> list[tuple[int, int]]:
    """Denoise a per-frame boolean "active" mask with a moving-average filter and
    return the (start, stop) index spans of the contiguous active stretches that
    are at least ``min_frames`` long."""
    is_active_moving_average = np.convolve(
        is_active_mask.astype(int),
        np.ones(filter_size) / filter_size,
        mode="same",
    )
    is_active_mask_filtered = is_active_moving_average > filtered_frac_threshold

    spans = []
    segment_start = None
    for i, is_active in enumerate(is_active_mask_filtered):
        if is_active and segment_start is None:
            segment_start = i
        elif not is_active and segment_start is not None:
            if i - segment_start >= min_frames:
                spans.append((segment_start, i))
            segment_start = None
    if segment_start is not None and len(is_active_mask_filtered) - segment_start >= min_frames:
        spans.append((segment_start, len(is_active_mask_filtered)))
    return spans


def classify_deeperfly_surface(points: np.ndarray, keypoint_names: list[str]) -> np.ndarray:
    """Classify, for each frame, which arena surface the fly is on
    ("floor", "lid" or "wall") from its 3D keypoints.

    The dataset's 3D is in the bottom (ventral) camera frame with +z pointing up
    (from the bottom camera toward the top). Following the dataset README, the
    body's dorsal direction is the cross product of the heading (hind to front
    thorax-coxa) and the right-to-left thorax-coxa direction. The fly is on the
    floor when its back points up, on the lid when it points down (the fly is
    upside down), and on a wall otherwise.

    Args:
        points: 3D keypoints of one animal, shape (n_frames, n_keypoints, 3), mm.
        keypoint_names: Names along the keypoint axis.

    Returns:
        Array of shape (n_frames,) of dtype str with values "floor", "lid" or
        "wall".
    """
    keypoint_index = {name: i for i, name in enumerate(keypoint_names)}
    front = points[:, [keypoint_index["lf_thorax_coxa"], keypoint_index["rf_thorax_coxa"]]].mean(1)
    hind = points[:, [keypoint_index["lh_thorax_coxa"], keypoint_index["rh_thorax_coxa"]]].mean(1)
    heading = front - hind
    heading = heading / np.linalg.norm(heading, axis=1, keepdims=True)
    left = points[:, [keypoint_index[f"l{pos}_thorax_coxa"] for pos in "fmh"]].mean(1)
    right = points[:, [keypoint_index[f"r{pos}_thorax_coxa"] for pos in "fmh"]].mean(1)
    right_to_left = left - right
    dorsal = np.cross(heading, right_to_left)
    dorsal = dorsal / np.linalg.norm(dorsal, axis=1, keepdims=True)
    dorsal_z = dorsal[:, 2]

    surface = np.full(dorsal_z.shape, "wall", dtype=object)
    surface[dorsal_z > DEEPERFLY_SURFACE_DORSAL_Z_THRESHOLD] = "floor"
    surface[dorsal_z < -DEEPERFLY_SURFACE_DORSAL_Z_THRESHOLD] = "lid"
    return surface.astype(str)


def read_deeperfly_fps(recording_path: str | Path) -> float:
    """Return the frame rate (frames per second) of a deeperfly HDF5 recording."""
    with h5py.File(Path(recording_path), "r") as recording_file:
        return float(recording_file.attrs["fps"])


def load_deeperfly_recording(
    recording_path: str | Path,
    min_duration_sec: float,
    surface: str = "any",
    filter_size: int = 5,
    filtered_frac_threshold: float = 0.5,
    moving_speed_threshold_mm_per_sec: float = DEEPERFLY_MOVING_SPEED_THRESHOLD_MM_PER_SEC,
    drop_initial_frames: int = DEEPERFLY_DROP_INITIAL_FRAMES,
    animal_index: int = 0,
) -> list[pd.DataFrame]:
    """Load a deeperfly HDF5 recording, split it into discrete moving segments,
    and return a list of DataFrames, each one being the joint-angle trajectory of
    a single segment.

    Each returned DataFrame has one row per video frame and one column per
    NeuroMechFly (flygym) joint angle, named exactly like the corresponding
    flygym JointDOF (e.g. "c_thorax-lf_coxa-yaw"), in radians. Two extra columns
    are added for inspection: "body_speed_mm_per_sec" and "surface". These
    DataFrames are meant to be passed to `simulate_one_segment` /
    `interpolate_trajectories` with ``source_format="deeperfly"``.

    Args:
        recording_path: Path to the deeperfly results HDF5 file
            (e.g. ".../deeperfly/results/r13_dense.h5").
        min_duration_sec: Minimum duration (seconds) a segment must have to be
            included.
        surface: Keep only frames where the fly is on this arena surface, one of
            "floor", "lid", "wall", or "any" to keep all surfaces. Freely-moving
            flies are often upside down on the lid; use "floor" to select
            floor-like walking kinematics.
        filter_size: Size of the moving-average filter used to denoise the
            moving/surface mask.
        filtered_frac_threshold: Threshold for the moving-average filter above
            which a frame is considered active (moving and on the chosen
            surface).
        moving_speed_threshold_mm_per_sec: Body-speed threshold above which the
            fly is considered to be moving.
        drop_initial_frames: Number of frames to drop from the start of the
            recording (the illumination is dark there; see the README).
        animal_index: Index along the animal axis (always 0 for single-fly
            recordings).

    Returns:
        A list of DataFrames, each containing a segment of moving kinematic
        trajectory as joint angles in the flygym convention.
    """
    if surface not in ("any", "floor", "lid", "wall"):
        raise ValueError(
            f"Unknown surface {surface!r}; expected 'any', 'floor', 'lid' or 'wall'."
        )

    recording_path = Path(recording_path)
    with h5py.File(recording_path, "r") as recording_file:
        frames_per_second = float(recording_file.attrs["fps"])
        keypoint_names = json.loads(recording_file.attrs["keypoints"])
        angle_names = json.loads(recording_file["ik/angles"].attrs["names"])
        joint_angles = recording_file["ik/angles"][animal_index]  # (n_frames, n_angles)
        keypoints_3d = recording_file["correction/points"][animal_index]  # (n_frames, n_kp, 3)

    joint_angles = joint_angles[drop_initial_frames:]
    keypoints_3d = keypoints_3d[drop_initial_frames:]

    # Body speed: the body position is the mean of the six thorax-coxa points,
    # and the speed is the frame-to-frame displacement times the frame rate.
    keypoint_index = {name: i for i, name in enumerate(keypoint_names)}
    thorax_coxa_indices = [keypoint_index[f"{leg}_thorax_coxa"] for leg in DEEPERFLY_LEGS]
    body_position = keypoints_3d[:, thorax_coxa_indices].mean(1)
    body_speed_mm_per_sec = np.concatenate(
        [[0.0], np.linalg.norm(np.diff(body_position, axis=0), axis=1)]
    ) * frames_per_second
    is_moving = body_speed_mm_per_sec > moving_speed_threshold_mm_per_sec

    surface_label = classify_deeperfly_surface(keypoints_3d, keypoint_names)
    if surface == "any":
        is_on_requested_surface = np.ones(len(surface_label), dtype=bool)
    else:
        is_on_requested_surface = surface_label == surface

    is_active_mask = is_moving & is_on_requested_surface
    min_frames = int(round(min_duration_sec * frames_per_second))
    spans = _split_active_mask_into_segments(
        is_active_mask, min_frames, filter_size, filtered_frac_threshold
    )

    segments = []
    for segment_start, segment_stop in spans:
        segment = pd.DataFrame(
            joint_angles[segment_start:segment_stop], columns=angle_names
        )
        segment["body_speed_mm_per_sec"] = body_speed_mm_per_sec[segment_start:segment_stop]
        segment["surface"] = surface_label[segment_start:segment_stop]
        segments.append(segment)

    return segments
