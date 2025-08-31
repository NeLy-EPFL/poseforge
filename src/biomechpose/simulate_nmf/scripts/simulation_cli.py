"""
CLI for simulation processing using tyro.
"""

import tyro
from pathlib import Path

from biomechpose.simulate_nmf.data import load_kinematic_recording
from biomechpose.simulate_nmf.simulate import simulate_one_segment
from biomechpose.simulate_nmf.postprocessing import postprocess_segment


def simulate_using_kinematic_prior(
    recorded_trial_path: str,  # path to a single .pkl file from Aymanns et al. 2022
    output_dir: str,  # output base directory
    input_timestep: float = 0.01,  # timestep of input kinematics (from Aymanns et al.)
    sim_timestep: float = 0.0001,  # timestep to use in the physics simulation
) -> None:
    recorded_trial_path = Path(recorded_trial_path)
    output_dir = Path(output_dir)
    assert (
        recorded_trial_path.is_file()
    ), f"Input path {recorded_trial_path} is not a file"
    print(f"Processing input path: {recorded_trial_path}")

    trial_name = recorded_trial_path.stem
    kinematic_recording_segments = load_kinematic_recording(
        recording_path=recorded_trial_path,
        min_duration_frames=10,
        filter_size=5,
        filtered_frac_threshold=0.5,
    )
    num_segments = len(kinematic_recording_segments)
    print(f"### Processing trial: {trial_name} ({num_segments} segments) ###")
    for segment_id, segment in enumerate(kinematic_recording_segments):
        print(f"=== Simulating segment {segment_id + 1}/{num_segments} ===")
        output_subdir = output_dir / trial_name / f"segment_{segment_id:03d}"
        simulate_one_segment(segment, output_subdir, input_timestep, sim_timestep)
        postprocess_segment(output_subdir, visualize=True)
    print(f"### Done processing trial: {trial_name} ###")


if __name__ == "__main__":
    tyro.cli(simulate_using_kinematic_prior)
