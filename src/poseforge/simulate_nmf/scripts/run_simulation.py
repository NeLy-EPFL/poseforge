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

from poseforge.simulate_nmf.data import load_kinematic_recording
from poseforge.simulate_nmf.simulate import simulate_one_segment
from poseforge.simulate_nmf.postprocessing import postprocess_segment
from poseforge.util import get_hardware_availability


def simulate_using_kinematic_prior(
    recorded_trial_path: str,
    trial_output_dir: str,
    segment_ids: list[int] | None = None,
    input_timestep: float = 0.01,
    sim_timestep: float = 0.0001,
    output_data_freq: int = 300,
    render_play_speed: float = 0.1,
    max_segments_per_trial: int | None = None,
    max_sim_steps_per_segment: int | None = None,
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
        output_data_freq (int): frequency (in Hz) to save data and render
            frames. Note that is not the frequency of the physics. It only
            affects the data saving rate. For example, if we want to match
            simulations to Spotlight recordings captured at 300 FPS, we
            should set this to 300.
        render_play_speed (float): Play speed to use when rendering the
            simulation. This affects neither the simulation itself nor the
            rendering and saving of data during the simulation. It only
            affects how the rendered video is played.
        max_segments_per_trial (int | None): If not None, limit the number
            of segments per trial to this number. This is mainly for
            testing.
        max_sim_steps_per_segment (int | None): If not None, for each
            segment to simulate, limit the number of simulation steps to
            this number. This is mainly for testing.
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
    if max_segments_per_trial is not None:
        kinematic_recording_segments = kinematic_recording_segments[
            :max_segments_per_trial
        ]
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
            kinematic_recording_segment=segment,
            output_dir=output_subdir,
            input_timestep=input_timestep,
            sim_timestep=sim_timestep,
            output_data_freq=output_data_freq,
            render_play_speed=render_play_speed,
            min_sim_duration_sec=0.2,
            max_sim_steps=max_sim_steps_per_segment,
        )
        if is_success:
            postprocess_segment(
                output_subdir, visualize=True, min_subsegment_duration_sec=0.1
            )
    print(f"### Done processing trial: {trial_name} ###")


def run_sequentially_for_testing():
    """Run everything sequentially (for debugging)"""
    # Configs
    output_basedir = Path("bulk_data/nmf_rendering_test/")
    input_timestep = 0.01
    sim_timestep = 0.0001
    trial_paths = [
        # For testing: change this list to limit the scope
        Path("bulk_data/kinematic_prior/aymanns2022/trials/BO_Gal4_fly1_trial001.pkl")
    ]

    # Limit scope of simulation as this is only for testing
    # Don't make `max_sim_steps_per_segment` too small; otherwise no subsegment-level
    # postprocessing will be performed
    max_segments_per_trial = 2
    max_sim_steps_per_segment = 3000

    # Process each trial
    for trial_path in trial_paths:
        simulate_using_kinematic_prior(
            trial_path,
            output_basedir / trial_path.stem,
            segment_ids=None,
            input_timestep=input_timestep,
            sim_timestep=sim_timestep,
            max_segments_per_trial=max_segments_per_trial,
            max_sim_steps_per_segment=max_sim_steps_per_segment,
        )


if __name__ == "__main__":
    import tyro

    get_hardware_availability(check_gpu=False, print_results=True)

    # Run the CLI
    tyro.cli(simulate_using_kinematic_prior)

    # # Run everything sequentially (for debugging)
    # run_sequentially_for_testing()
