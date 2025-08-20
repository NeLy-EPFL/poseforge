import numpy as np
import pandas as pd
import dm_control.mujoco
from tqdm import trange
from pathlib import Path
from dm_control.rl.control import PhysicsError
from flygym import Fly, SingleFlySimulation, Camera
from flygym.arena import BaseArena, FlatTerrain
from flygym.preprogrammed import all_leg_dofs

from biomechpose.simulate_nmf.data import interpolate_trajectories
from biomechpose.simulate_nmf.utils import parse_nmf_joint_name, parse_nmf_keypoint_name


class SpotlightArena(FlatTerrain):
    def __init__(
        self,
        size: tuple[float, float] = (100, 100),
        friction: tuple[float, float, float] = (1, 0.005, 0.0001),
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


def set_up_simulation(render_window_size, render_play_speed, render_fps, sim_timestep):
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

    return sim, camera, dm_camera, keypoint_names, bound_keypoint_position_sensors


def run_neuromechfly_simulation(
    trajectories_interp, render_window_size, render_play_speed, render_fps, sim_timestep
):
    sim, camera, dm_camera, keypoint_names, bound_keypoint_position_sensors = (
        set_up_simulation(
            render_window_size, render_play_speed, render_fps, sim_timestep
        )
    )

    timestamps_hist = []
    joints_angles_hist = []
    keypoints_pos_world_hist = []
    keypoints_pos_cam_hist = []
    cardinal_vectors_hist = []

    camera_matrix_hist = []  # see https://en.wikipedia.org/wiki/Camera_matrix

    # Simulation loop
    last_num_frames = 0
    for sim_frame_id in trange(trajectories_interp.shape[0], disable=None):
        # if sim_frame_id == 1000: break  # For debugging, remove later
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

        timestamps_hist.append(sim.curr_time)

        # Store joint angles
        joints_angles_hist.append(observation["joints"][0, :].copy())

        # Store keypoint positions
        camera_matrix = dm_camera.matrix.copy()
        keypoints_pos_world = (
            bound_keypoint_position_sensors.sensordata.copy().reshape((-1, 3)).T
        )
        keypoints_pos_world_homogeneous = np.vstack(
            [keypoints_pos_world, np.ones((1, keypoints_pos_world.shape[1]))]
        )
        keypoints_pos_cam = camera_matrix @ keypoints_pos_world_homogeneous
        keypoints_pos_cam_hist.append(keypoints_pos_cam)

        fly_center_of_mass_pos = observation["fly"][0, :]
        keypoints_pos_world_centered = (
            keypoints_pos_world - fly_center_of_mass_pos[:, np.newaxis]
        )
        keypoints_pos_world_hist.append(keypoints_pos_world_centered)

        # Store forward vector
        cardinal_vectors_hist.append(observation["cardinal_vectors"].copy())

        # Store camera matrix
        camera_matrix_hist.append(camera_matrix)

        # Save rendered image
        last_num_frames = len(camera._frames)

        # Check if the fly has flipped over
        if observation["cardinal_vectors"][2, 2] < 0:
            print(f"Fly flipped over at frame {sim_frame_id}. Stopping simulation.")
            break

    return (
        camera,
        timestamps_hist,
        joints_angles_hist,
        keypoints_pos_world_hist,
        keypoints_pos_cam_hist,
        cardinal_vectors_hist,
        camera_matrix_hist,
        keypoint_names,
    )


def make_kinematic_states_dataframe(
    timestamps_hist,
    joints_angles_hist,
    keypoints_pos_world_hist,
    keypoints_pos_cam_hist,
    cardinal_vectors_hist,
    camera_matrix_hist,
    keypoint_names,
):
    joints_angles_hist = np.array(joints_angles_hist)  # (len, n_dofs)
    keypoints_pos_world_hist = np.array(keypoints_pos_world_hist)  # (len, 3, n_keypts)
    keypoints_pos_cam_hist = np.array(keypoints_pos_cam_hist)  # (len, 3, n_keypts)
    cardinal_vectors_hist = np.array(cardinal_vectors_hist)  # (len, 3, 3(xyz))

    columns = {}
    columns["time"] = timestamps_hist

    for i, dof_name in enumerate(all_leg_dofs):
        leg, canonical_dof_name = parse_nmf_joint_name(dof_name)
        key = leg + canonical_dof_name
        columns[f"dof_angle_{key}"] = joints_angles_hist[:, i]

    for i, keypoint_name in enumerate(keypoint_names):
        leg, canonical_keypoint_name = parse_nmf_keypoint_name(keypoint_name)
        key = leg + canonical_keypoint_name
        columns[f"keypoint_pos_world_{key}_x"] = keypoints_pos_world_hist[:, 0, i]
        columns[f"keypoint_pos_world_{key}_y"] = keypoints_pos_world_hist[:, 1, i]
        columns[f"keypoint_pos_world_{key}_z"] = keypoints_pos_world_hist[:, 2, i]

    for i, keypoint_name in enumerate(keypoint_names):
        leg, canonical_keypoint_name = parse_nmf_keypoint_name(keypoint_name)
        key = leg + canonical_keypoint_name
        columns[f"keypoint_pos_cam_{key}_col"] = keypoints_pos_cam_hist[:, 0, i]
        columns[f"keypoint_pos_cam_{key}_row"] = keypoints_pos_cam_hist[:, 1, i]
        columns[f"keypoint_pos_cam_{key}_depth"] = keypoints_pos_cam_hist[:, 2, i]

    columns["cardinal_vector_forward"] = list(np.array(cardinal_vectors_hist)[:, 0, :])
    columns["cardinal_vector_left"] = list(np.array(cardinal_vectors_hist)[:, 1, :])
    columns["cardinal_vector_up"] = list(np.array(cardinal_vectors_hist)[:, 2, :])
    kinematic_states_df = pd.DataFrame(columns, dtype=np.float32)

    # Add camera matrix to the DataFrame. Do this after creating the DataFrame with
    # dtype=float32 for all the other columns because unlike others, the camera matrix
    # column contains 3x4 matrices, which cannot be casted to float32.
    kinematic_states_df["camera_matrix"] = camera_matrix_hist

    kinematic_states_df.index.name = "frame_id"
    return kinematic_states_df


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
        timestamps_hist,
        joints_angles_hist,
        keypoints_pos_world_hist,
        keypoints_pos_cam_hist,
        cardinal_vectors_hist,
        camera_matrix_hist,
        keypoint_names,
    ) = run_neuromechfly_simulation(
        trajectories_interp,
        render_window_size,
        render_play_speed,
        render_fps,
        sim_timestep,
    )

    # Save kinematic states as Pandas DataFrame
    kinematic_states_df = make_kinematic_states_dataframe(
        timestamps_hist,
        joints_angles_hist,
        keypoints_pos_world_hist,
        keypoints_pos_cam_hist,
        cardinal_vectors_hist,
        camera_matrix_hist,
        keypoint_names,
    )
    kinematic_states_df.to_pickle(output_dir / "kinematic_states_history.pkl")

    # Save rendered frames as a video
    camera.save_video(output_dir / "simulation_rendering.mp4", stabilization_time=0)
