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

Terminology:
- Kinematic recording: A single train from Aymanns et al. 2022.
- Segment: A continuous period within a kinematic recording where the fly
  is NOT as rest, as indicated by the behavior classification in Aymanns et
  al. There can be multiple segments in a single kinematic recording.
- Subsegment (see postprocessing): A continuous period within a segment
  where the fly is upright (defined using a maximum allowed tilting angle).
  The subsegments can only be identified AFTER the physics simulation
  because we don't know if the fly will flip upside down until we actually
  simulate it.

Pipeline:
- First copy & extract kinematic recordings from Aymanns et al. 2022 (this
  is done separately in copy_kinematic_recording.py).
- Run this script to:
    - Load the kinematic recording.
    - Split it into segments based on metadata from Aymanns et al.
    - Run the physics simulation for each segment. This generates the
      history of the fly's kinematic states in the simulation (but, for
      example, in world coordinates) and renders a video looking at the fly
      from the bottom up.
    - Postprocess the simulation results to:
        - Split each segment into subsegments based on the fly's
          orientation
        - For each subsegment, center keypoint coordinates around the fly's
          center of mass and rotate the rendered frame so that the fly's
          anterior-posterior axis is aligned with the vertical axis with
          the head facing up.
        - Optionally save a visualization of each subsegment as a video.
"""

from pathlib import Path

from biomechpose.simulate_nmf.data import load_kinematic_recording
from biomechpose.simulate_nmf.simulate import simulate_one_segment
from biomechpose.simulate_nmf.postprocessing import postprocess_segment


if __name__ == "__main__":
    kinematic_recording_dir = Path("bulk_data/kinematic_prior/aymanns2022/trials/")
    output_dir = Path("bulk_data/nmf_rendering_seglabels/")
    input_timestep = 0.01
    sim_timestep = 0.0001
    max_segments = 1000  # limit to this many segments per trial

    # trial_paths = sorted(list(kinematic_recording_dir.glob("*.pkl")))
    trial_paths = [
        Path("bulk_data/kinematic_prior/aymanns2022/trials/BO_Gal4_fly1_trial001.pkl")
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
            print(f"=== Simulating segment {segment_id + 1}/{num_segments} ===")
            output_subdir = output_dir / trial_name / f"segment_{segment_id:03d}"
            simulate_one_segment(segment, output_subdir, input_timestep, sim_timestep)
            postprocess_segment(output_subdir, visualize=True)
