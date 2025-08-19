"""
This script replays the kinematics of fly behavior from Aymanns et al.
2022) in FlyGym. It interpolates the recorded kinematics to match the
NeuroMechFly simulation timestep, runs the simulation with position
control, and saves both rendered images and kinematic states histories.

Outputs for each simulated segment, the script produces a directory
containing:
- Video of the simulation (simulation_rendering.mp4).
- Kinematic states history as a Pandas DataFrame (kinematic_states_history.pkl).

The kinematic states DataFrame contains:
    - "time": Simulation time (seconds) for each frame.
    - "dof_angle_{key}": Joint angle (radians) for each DoF.
    - "keypoint_pos_3d_{key}_{x,y,z}": 3D position (in camera coordinates,
          i.e. x, y, depth) of each keypoint.
    - "forward_vector_{x,y,z}": Components of the fly's forward vector
          (direction on that the fly is facing).
    - "camera_matrix": 3x4 camera matrix (numpy array) for each frame
          (see https://en.wikipedia.org/wiki/Camera_matrix).
"""

import pandas as pd
from pathlib import Path

from biomechpose.simulate_nmf.data import (
    load_kinematic_recording,
    interpolate_trajectories,
)
from biomechpose.simulate_nmf.simulate import simulate, make_kinematic_states_dataframe


def simulate_one_segment(
    kinematic_recording_segment: pd.DataFrame,
    output_dir: Path,
    input_timestep: float,
    sim_timestep: float,
    render_fps: int = 300,
    render_play_speed: float = 0.1,
    render_window_size=(720, 720),
) -> None:
    """Simulate a single segment of kinematic recording in FlyGym."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Interpolate the trajectories to match the simulation timestep
    trajectories_interp, interp_factor = interpolate_trajectories(
        kinematic_recording_segment, input_timestep, sim_timestep
    )
    print(f"Simulating segment with {len(kinematic_recording_segment)} frames.")
    print(f"Interpolation factor: {interp_factor}")
    print(f"Output directory: {output_dir}")

    # Run the simulation
    (
        camera,
        joints_angles_hist,
        keypoints_pos_world_hist,
        keypoints_pos_cam_hist,
        cardinal_vectors_hist,
        camera_matrix_hist,
        keypoint_names,
    ) = simulate(
        trajectories_interp,
        render_window_size,
        render_play_speed,
        render_fps,
        sim_timestep,
    )

    # Save kinematic states as Pandas DataFrame
    kinematic_states_df = make_kinematic_states_dataframe(
        joints_angles_hist,
        keypoints_pos_world_hist,
        keypoints_pos_cam_hist,
        cardinal_vectors_hist,
        camera_matrix_hist,
        sim_timestep,
        keypoint_names,
    )
    kinematic_states_df.to_pickle(output_dir / "kinematic_states_history.pkl")

    # Save rendered frames as a video
    camera.save_video(output_dir / "simulation_rendering.mp4", stabilization_time=0)


if __name__ == "__main__":
    kinematic_recording_dir = Path("bulk_data/kinematic_recording/aymanns2022/trials/")
    output_dir = Path("bulk_data/nmf_rendering")
    input_timestep = 0.01
    sim_timestep = 0.0001
    max_segments = 1  # limit to this many segments per trial

    # trial_paths = sorted(list(kinematic_recording_dir.glob("*.pkl")))
    trial_paths = [
        Path(
            "bulk_data/kinematic_recording/aymanns2022/trials/BO_Gal4_fly1_trial001.pkl"
        )
    ]

    for trial_path in trial_paths:
        trial_name = trial_path.stem
        kinematic_recording_segments = load_kinematic_recording(
            recording_path=trial_path,
            min_duration_frames=10,
            filter_size=5,
            filtered_frac_threshold=0.5,
        )
        kinematic_recording_segments = kinematic_recording_segments[:max_segments]
        num_segments = len(kinematic_recording_segments)
        print(f"### Processing trial: {trial_name} ({num_segments} segments) ###")
        for segment_id, segment in enumerate(kinematic_recording_segments):
            print(f"Simulating segment {segment_id + 1}/{num_segments}")
            output_subdir = output_dir / trial_name / f"segment_{segment_id}"
            simulate_one_segment(segment, output_subdir, input_timestep, sim_timestep)
