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
    trial_output_dir: str,  # output base directory
    segment_ids: list[int] | None = None,  # segments to simulate; if None simulate all
    input_timestep: float = 0.01,  # timestep of input kinematics (from Aymanns et al.)
    sim_timestep: float = 0.0001,  # timestep to use in the physics simulation
) -> None:
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
        min_duration_frames=10,
        filter_size=5,
        filtered_frac_threshold=0.5,
    )
    num_segments = len(kinematic_recording_segments)
    if segment_ids is None:
        segment_ids = list(range(num_segments))
    else:
        for segid in segment_ids:
            assert (
                0 <= segid < num_segments
            ), f"Invalid segment_id {segid} for trial with {num_segments} segments"

    print(f"### Processing trial: {trial_name} ({num_segments} segments) ###")
    # for segment_id, segment in enumerate(kinematic_recording_segments):
    for segid in segment_ids:
        print(f"=== Simulating segment {segid} (out of {num_segments} segments) ===")
        segment = kinematic_recording_segments[segid]
        output_subdir = trial_output_dir / f"segment_{segid:03d}"
        simulate_one_segment(segment, output_subdir, input_timestep, sim_timestep)
        postprocess_segment(output_subdir, visualize=True)
    print(f"### Done processing trial: {trial_name} ###")


if __name__ == "__main__":
    tyro.cli(simulate_using_kinematic_prior)
