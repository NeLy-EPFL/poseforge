"""
This script replays the kinematics of fly behavior from Aymanns et al.
2022) in FlyGym. It interpolates the recorded kinematics to match the
NeuroMechFly simulation timestep, runs the simulation with position
control, and saves both rendered images and kinematic states histories.

Outputs for each simulated segment, the script produces a directory
containing:
- Rendered frames as PNG images (frame_XXXX.png).
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

from typing import Optional
import numpy as np
import pandas as pd
import dm_control.mujoco
from pathlib import Path
from PIL import Image
from tqdm import trange, tqdm
from dm_control.rl.control import PhysicsError
from flygym import Fly, SingleFlySimulation, Camera
from flygym.arena import BaseArena, FlatTerrain
from flygym.state import KinematicPose
from flygym.preprogrammed import all_leg_dofs

from biomechpose.simulate_nmf.utils import parse_nmf_joint_name, parse_nmf_keypoint_name


def recording_row_to_pose(recording_row: pd.Series) -> KinematicPose:
    """Generate FlyGym KinematicPose object from a row of kinematic
    recording from Aymanns et al. 2022, as processed by the
    copy_kinematic_recording.py script."""
    joint_angles = {}
    for nmf_joint_name in all_leg_dofs:
        leg, aymanns_dof_name = parse_nmf_joint_name(nmf_joint_name)
        joint_angle = recording_row[f"Angle__{leg}_leg_{aymanns_dof_name}"]
        joint_angles[nmf_joint_name] = joint_angle
    return KinematicPose(joint_angles)


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


def get_keypoint_position_sensors(fly: Fly):
    """Create a list of keypoint position sensors, one for each keypoint to
    be tracked by the pose estimation model.
    This is a bit of a hack: the claws already have their own position
    sensors (these are the end effector position sensors built into
    FlyGym), so we simply insert them to the sensors list at the end. Do
    it in reverse order to avoid messing up the indices."""
    keypoint_position_names = []
    keypoint_position_sensors = []
    all_leg_names = [f"{side}{pos}" for side in "LR" for pos in "FMH"]
    for i, leg in enumerate(all_leg_names):
        # Add keypoint position sensors for each link except the claws
        # (which are already handled by FlyGym's end effector sensors)
        for keypoint in ["Coxa", "Femur", "Tibia", "Tarsus1"]:
            keypoint_name = f"{leg}{keypoint}"
            keypoint_position_names.append(keypoint_name)
            keypoint_position_sensors.append(
                fly.model.sensor.add(
                    "framepos",
                    name=f"{keypoint_name}_pos",
                    objtype="xbody",
                    objname=keypoint_name,
                )
            )
        # Add the claws position sensors which are already initialized as
        # end effector position sensors
        keypoint_position_names.append(f"{leg}Tarsus5")
        keypoint_position_sensors.append(fly._end_effector_sensors[i])

    return keypoint_position_names, keypoint_position_sensors


class SpotlightArena(FlatTerrain):
    def __init__(
        self,
        size: tuple[float, float] = (100, 100),
        friction: tuple[float, float, float] = (1, 0.005, 0.0001),
        scale_bar_pos: Optional[tuple[float, float, float]] = None,
    ):
        BaseArena.__init__(self)

        ground_size = [*size, 1]
        self.root_element.worldbody.add(
            "geom",
            type="plane",
            name="ground",
            size=ground_size,
            friction=friction,
            conaffinity=0,
            rgba=(1, 1, 1, 0),
        )
        self.root_element.worldbody.add(
            "geom",
            type="box",  # somehow plane doesn't work
            name="background",
            size=(100, 100, 1),
            pos=(0, 0, 5),
            friction=friction,
            conaffinity=0,
            rgba=(0, 0, 0, 1),
        )
        self.friction = friction

    def get_spawn_position(self, rel_pos, rel_angle):
        return rel_pos, rel_angle


def load_kinematic_recording(
    recording_path: str | Path,
    min_duration_frames: int,
    filter_size: int = 5,
    filtered_frac_threshold: float = 0.5,
) -> list[pd.DataFrame]:
    """Load kinematic recording from a PKL file, split it into discrete,
    non-resting segments, and return a list of DataFrames, each one being
    the trajectory of a single segment.

    Args:
        recording_path: Path to the kinematic recording PKL file.
        min_duration_frames: Minimum number of frames a segment must have
            to be included.
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
            if i - segment_start >= min_duration_frames:
                segments.append(kinematic_recording.iloc[segment_start:i])
            segment_start = None

    return segments


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
    interp_factor = input_timestep / sim_timestep
    if int(interp_factor) != interp_factor:
        raise ValueError(
            f"Input timestep {input_timestep} must be a multiple of simulation timestep {sim_timestep}."
        )
    interp_factor = int(interp_factor)
    print(f"Simulating segment with {len(kinematic_recording_segment)} frames.")
    print(f"Interpolation factor: {interp_factor}")
    print(f"Output directory: {output_dir}")
    trajectories_interp = extract_joint_angles_trajectory(
        kinematic_recording_segment, interp_factor
    )

    # Set up simulation
    fly = Fly(
        init_pose="stretch",
        control="position",
        actuated_joints=all_leg_dofs,
        xml_variant="deepfly3d",
        spawn_pos=(0, 0, 0.5),
        joint_stiffness=0.1,
    )
    keypoint_names, keypoint_position_sensors = get_keypoint_position_sensors(fly)
    camera_params = {
        "mode": "track",
        "pos": (0, 0, -100),
        "euler": (0, np.pi, -np.pi / 2),
        "fovy": 5,
    }
    camera = Camera(
        attachment_point=fly.model.worldbody,
        camera_name=f"flytrack_cam",
        camera_parameters=camera_params,
        window_size=render_window_size,
        play_speed=render_play_speed,
        fps=render_fps * render_play_speed,
        draw_contacts=False,
        play_speed_text=False,
    )
    sim = SingleFlySimulation(
        fly=fly,
        cameras=[camera],
        arena=SpotlightArena(),
        timestep=sim_timestep,
    )
    dm_camera = dm_control.mujoco.Camera(
        sim.physics,
        camera_id=camera.camera_id,
        width=camera.window_size[0],
        height=camera.window_size[1],
    )
    bound_keypoint_position_sensors = sim.physics.bind(keypoint_position_sensors)

    joints_angles_hist = []
    keypoints_pos_3d_hist = []
    # keypoints_pos_2d_hist = []
    forward_vector_hist = []
    camera_matrix_hist = []  # see https://en.wikipedia.org/wiki/Camera_matrix

    # Simulation loop
    last_num_frames = 0
    for sim_frame_id in trange(trajectories_interp.shape[0], disable=None):
        # if sim_frame_id == 1000: break  # For testing, limit to 1000 frames
        action = {"joints": trajectories_interp[sim_frame_id]}
        try:
            observation, reward, terminated, truncated, info = sim.step(action)
        except PhysicsError as e:
            print(f"Simulation error at frame {sim_frame_id}: {e}")
            break
        rendered_image = sim.render()[0]
        
        # Check if a new frame is rendered. If not, continue without extracting any data
        if not len(camera._frames) > last_num_frames:
            continue

        # Store joint angles
        joints_angles_hist.append(observation["joints"][0, :].copy())

        # Store keypoint positions
        camera_matrix = dm_camera.matrix.copy()
        keypoints_pos_3d_global = (
            bound_keypoint_position_sensors.sensordata.copy().reshape((-1, 3)).T
        )
        keypoints_pos_3d_global_homogeneous = np.vstack(
            [keypoints_pos_3d_global, np.ones((1, keypoints_pos_3d_global.shape[1]))]
        )
        keypoints_pos_3d_cam = camera_matrix @ keypoints_pos_3d_global_homogeneous
        keypoints_pos_3d_cam = keypoints_pos_3d_cam / keypoints_pos_3d_cam[2]
        keypoints_pos_3d_hist.append(keypoints_pos_3d_cam)
        # keypoints_pos_2d_hist.append(keypoints_pos_2d_cam)

        # Store forward vector
        forward_vector = observation["cardinal_vectors"][0, :].copy()
        forward_vector_hist.append(forward_vector)

        # Store camera matrix
        camera_matrix_hist.append(camera_matrix)

        # Save rendered image
        last_num_frames = len(camera._frames)
        # image_path = output_dir / f"frame_{sim_frame_id:04d}.png"
        # Image.fromarray(rendered_image).save(image_path)

    # Save kinematic states as Pandas DataFrame
    joints_angles_hist = np.array(joints_angles_hist)  # (n_frames, n_dofs)
    keypoints_pos_3d_hist = np.array(keypoints_pos_3d_hist)  # (n_frames, 3, n_keypts)
    # keypoints_pos_2d_hist = np.array(keypoints_pos_2d_hist)  # (n_frames, 2, n_keypts)
    forward_vector_hist = np.array(forward_vector_hist)  # (n_frames, 3)
    columns = {}
    columns["time"] = np.arange(len(joints_angles_hist)) * sim_timestep
    for i, dof_name in enumerate(all_leg_dofs):
        leg, canonical_dof_name = parse_nmf_joint_name(dof_name)
        key = leg + canonical_dof_name
        columns[f"dof_angle_{key}"] = joints_angles_hist[:, i]
    for i, keypoint_name in enumerate(keypoint_names):
        leg, canonical_keypoint_name = parse_nmf_keypoint_name(keypoint_name)
        key = leg + canonical_keypoint_name
        columns[f"keypoint_pos_3d_{key}_x"] = keypoints_pos_3d_hist[:, 0, i]
        columns[f"keypoint_pos_3d_{key}_y"] = keypoints_pos_3d_hist[:, 1, i]
        columns[f"keypoint_pos_3d_{key}_z"] = keypoints_pos_3d_hist[:, 2, i]
    # for i, keypoint_name in enumerate(keypoint_names):
    #     leg, canonical_keypoint_name = parse_nmf_keypoint_name(keypoint_name)
    #     key = leg + canonical_keypoint_name
    #     columns[f"keypoint_pos_2d_{key}_x"] = keypoints_pos_2d_hist[:, 0, i]
    #     columns[f"keypoint_pos_2d_{key}_y"] = keypoints_pos_2d_hist[:, 1, i]
    columns["forward_vector_x"] = np.array(forward_vector_hist)[:, 0]
    columns["forward_vector_y"] = np.array(forward_vector_hist)[:, 1]
    columns["forward_vector_z"] = np.array(forward_vector_hist)[:, 2]
    kinematic_states_df = pd.DataFrame(columns, dtype=np.float32)
    # Add camera matrix to the DataFrame. Do this after creating the DataFrame with
    # dtype=float32 for all the other columns because unlike others, the camera matrix
    # column contains 3x4 matrices, which cannot be casted to float32.
    kinematic_states_df["camera_matrix"] = camera_matrix_hist
    kinematic_states_df.index.name = "frame_id"
    kinematic_states_df.to_pickle(output_dir / "kinematic_states_history.pkl")
    
    # Save rendered frames as a video
    camera.save_video(output_dir / "simulation_rendering.mp4")


if __name__ == "__main__":
    kinematic_recording_dir = Path("bulk_data/kinematic_recording/aymanns2022/trials/")
    output_dir = Path("bulk_data/nmf_rendering")
    input_timestep = 0.01
    sim_timestep = 0.0001
    max_segments = 2  # limit to this many segments per trial

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
