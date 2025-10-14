import numpy as np
import pandas as pd
from pathlib import Path
from flygym.preprogrammed import all_leg_dofs

from poseforge.neuromechfly.constants import parse_nmf_joint_name


def extract_joint_angles_trajectory(
    kinematic_recording_segment: pd.DataFrame, interp_factor: int
) -> np.ndarray:
    """Extract joint angles trajectory from a kinematic recording segment,
    interpolating the angles to match the simulation timestep."""
    # Extract the original joint angles from the kinematic recording segment
    original_trajectories_dict = {}
    for nmf_joint_name in all_leg_dofs:
        leg, aymanns_dof_name = parse_nmf_joint_name(nmf_joint_name)
        column_name = f"Angle__{leg}_leg_{aymanns_dof_name}"
        time_series = kinematic_recording_segment[column_name].values
        original_trajectories_dict[nmf_joint_name] = time_series
    original_trajectories_array = np.array(
        [original_trajectories_dict[dof] for dof in all_leg_dofs]
    ).T

    # Interpolate the joint angles to match the simulation timestep
    num_frames = len(kinematic_recording_segment)
    interp_num_frames = num_frames * interp_factor
    interp_trajectories_array = np.zeros((interp_num_frames, len(all_leg_dofs)))
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
):
    interp_factor = input_timestep / sim_timestep
    if int(interp_factor) != interp_factor:
        raise ValueError(
            f"Input timestep {input_timestep} must be a multiple of "
            f"simulation timestep {sim_timestep}."
        )
    interp_factor = int(interp_factor)
    trajectories_interp = extract_joint_angles_trajectory(
        kinematic_recording_segment, interp_factor
    )
    return trajectories_interp, interp_factor
