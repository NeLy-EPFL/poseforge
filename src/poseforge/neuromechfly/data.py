import numpy as np
import pandas as pd
from pathlib import Path

from poseforge.neuromechfly.constants import parse_nmf_joint #, all_leg_dofs
from flygym.anatomy import JointDOF

def extract_joint_angles_trajectory(
    kinematic_recording_segment: pd.DataFrame, interp_factor: int, actuated_joints_list: list[JointDOF], use_flybody: bool = False
) -> np.ndarray:
    """Extract joint angles trajectory from a kinematic recording segment,
    interpolating the angles to match the simulation timestep."""
    # Extract the original joint angles from the kinematic recording segment
    original_trajectories_list = []
    n_joints = 0
    for joint in actuated_joints_list:
        leg, aymans_dof = parse_nmf_joint(joint)
        column_name = f"Angle__{leg}_leg_{aymans_dof}"
        time_series = kinematic_recording_segment[column_name].values 
        if not use_flybody:
            if leg.startswith("R"):
                if "roll" in aymans_dof:
                    time_series = -time_series  # Negate roll angles for right legs to match NMF convention
                if "yaw" in aymans_dof:
                    time_series = -time_series  # Negate angles for right legs to match NMF convention
            
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
):
    interp_factor = input_timestep / sim_timestep
    if int(interp_factor) != interp_factor:
        raise ValueError(
            f"Input timestep {input_timestep} must be a multiple of "
            f"simulation timestep {sim_timestep}."
        )
    interp_factor = int(interp_factor)
    trajectories_interp = extract_joint_angles_trajectory(
        kinematic_recording_segment, interp_factor, actuated_joints_list, use_flybody
    )
    return trajectories_interp, interp_factor
