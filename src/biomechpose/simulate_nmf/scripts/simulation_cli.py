"""
CLI for simulation processing using tyro.
"""

import tyro
from pathlib import Path

from biomechpose.simulate_nmf.data import load_kinematic_recording
from biomechpose.simulate_nmf.simulate import simulate_one_segment
from biomechpose.simulate_nmf.postprocessing import postprocess_segment
from biomechpose.util import print_hardware_availability


def simulate_using_kinematic_prior(
    recorded_trial_path: str,
    trial_output_dir: str,
    segment_ids: list[int] | None = None,
    input_timestep: float = 0.01,
    sim_timestep: float = 0.0001,
) -> None:
    """CLI to run replay kinematic motion priors from Aymanns et al. 2022
    in NeuroMechFly.

    Args:
        recorded_trial_path (str): Path to a single .pkl file from Aymanns
            et al. 2022.
        trial_output_dir (str): Output base directory.
        segment_ids (list[int] | None): List of segment IDs to simulate. If
            None, simulate all segments in the trial.
        input_timestep (float): Timestep of input kinematics (from Aymanns
            et al. 2022).
        sim_timestep (float): Timestep to use in the physics simulation.
    """
    recorded_trial_path = Path(recorded_trial_path)
    trial_output_dir = Path(trial_output_dir)
    assert (
        recorded_trial_path.is_file()
    ), f"Input path {recorded_trial_path} is not a file"
    print(
        f"Processing input path: {recorded_trial_path}, "
        f"segment_ids: {segment_ids if segment_ids is not None else 'all'}"
    )

    trial_name = recorded_trial_path.stem
    kinematic_recording_segments = load_kinematic_recording(
        recording_path=recorded_trial_path,
        min_duration_sec=0.2,
        input_timestep=input_timestep,
        filter_size=5,
        filtered_frac_threshold=0.5,
    )
    num_segments = len(kinematic_recording_segments)
    if segment_ids is None:
        segment_ids = list(range(num_segments))
    else:
        for segment_id in segment_ids:
            assert (
                0 <= segment_id < num_segments
            ), f"Invalid segment_id {segment_id} for trial with {num_segments} segments"

    print(f"### Processing trial: {trial_name} ({num_segments} segments) ###")
    for segment_id in segment_ids:
        print(f"=== Simulating segment #{segment_id} ({num_segments} total) ===")
        segment = kinematic_recording_segments[segment_id]
        output_subdir = trial_output_dir / f"segment_{segment_id:03d}"
        is_success = simulate_one_segment(
            segment,
            output_subdir,
            input_timestep,
            sim_timestep,
            min_sim_duration_sec=0.2,
        )
        if is_success:
            postprocess_segment(
                output_subdir, visualize=True, min_subsegment_duration_sec=0.1
            )
    print(f"### Done processing trial: {trial_name} ###")


if __name__ == "__main__":
    print_hardware_availability(check_gpu=False)
    tyro.cli(simulate_using_kinematic_prior)
