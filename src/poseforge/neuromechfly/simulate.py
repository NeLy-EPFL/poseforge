import numpy as np
import pandas as pd
import imageio
import h5py
from collections import defaultdict
from tqdm import trange
from pathlib import Path

from flygym.compose import Fly, KinematicPosePreset, ActuatorType
from flygym.anatomy import Skeleton, AxisOrder, JointPreset, ActuatedDOFPreset

from flygym.anatomy import JointDOF, RotationAxis, BodySegment
from flygym.compose import FlatGroundWorld, TetheredWorld
from flygym.utils.math import Rotation3D
from flygym import Simulation, Renderer

import mujoco as mj
import dm_control.mjcf as mjcf
import numpy as np
from jaxtyping import Float

from poseforge.neuromechfly.data import interpolate_trajectories
from poseforge.neuromechfly.constants import (
    parse_nmf_joint_name,
    # color_by_link,
    # color_by_kinematic_chain,
    # color_palette,
)

axis_order = AxisOrder.ROLL_YAW_PITCH
articulated_joints = JointPreset.LEGS_ONLY
actuated_dofs = ActuatedDOFPreset.LEGS_ACTIVE_ONLY
neutral_pose = KinematicPosePreset.NEUTRAL
actuator_type = ActuatorType.POSITION
skeleton = Skeleton(axis_order=axis_order, joint_preset=articulated_joints)


class SpotlightArena(TetheredWorld):
    def __init__(
        self,
        name: str = "spotlight_arena",
        size: tuple[float, float] = (100, 100),
        friction: tuple[float, float, float] = (1, 0.005, 0.0001),
    ):
        super().__init__(name)
        #self.ground_geom.remove() # remove the ground with texture add transparent floor

        ground_size = [*size, 1]
        self.ground_geom = self.mjcf_root.worldbody.add(
            "geom",
            type="plane",
            name="ground",
            size=ground_size,
            friction=friction,
            conaffinity=0,
            rgba=(1, 1, 1, 0),
        )
        self.friction = friction

        # # Remove lights
        # for light in self.root_element.root.worldbody.light:
        #     light.remove()

        # # Make lights non-shadow-casting to make segments look more uniform
        # self.root_element.default.light.castshadow = False

        self.legpos_to_groundcontactsensors_by_fly = None

    def get_spawn_position(self, rel_pos, rel_angle):
        return rel_pos, rel_angle


class FlyForRendering(Fly):
    def __init__(self, visual_paths:list[Path], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_body_segments_names = [
            b.name for b in self.get_bodysegs_order()
        ]
        self.visual_paths = visual_paths
        self._add_mesh_state_sensors()
        self._add_cardinal_sensors()
        self.current_color_coding: int = -1
    
    def colorize(self, next_color_coding_idx: int):
        if next_color_coding_idx == self.current_color_coding:
            return
        super().colorize(self.visual_paths[next_color_coding_idx])
        self.current_color_coding = next_color_coding_idx
        return

    def _add_pos_and_quat_sensors(self, segment_name):
        sensors = {}

        # Position in body frame (xbody), i.e. at joint with parent body
        sensors["pos_atparent"] = self.mjcf_root.sensor.add(
            "framepos",
            name=f"{segment_name}_pos_atparent",
            objtype="geom",
            objname=segment_name,
        )

        # The following sensors are not currently used, so we comment them out to
        # avoid cluttering the simulation with too many sensors.
        # # Rotation in body frame (xbody), i.e. at joint with parent body
        # sensors["quat_atparent"] = self.mjcf_root.sensor.add(
        #     "framequat",
        #     name=f"{segment_name}_quat_atparent",
        #     objtype="xbody",
        #     objname=segment_name,
        # )

        # # Position in inertial frame (body), i.e. at COM of body
        # sensors["pos_com"] = self.mjcf_root.sensor.add(
        #     "framepos",
        #     name=f"{segment_name}_pos_com",
        #     objtype="body",
        #     objname=segment_name,
        # )

        # # Rotation in inertial frame (body), i.e. at COM of body
        # sensors["quat_com"] = self.mjcf_root.sensor.add(
        #     "framequat",
        #     name=f"{segment_name}_quat_com",
        #     objtype="body",
        #     objname=segment_name,
        # )

        return sensors

    def _add_mesh_state_sensors(self):
        self.body_segment_sensor_lookup = defaultdict(dict)
        for segment_name in self.all_body_segments_names:
            sensors = self._add_pos_and_quat_sensors(segment_name)
            for sensor_type, sensor_obj in sensors.items():
                self.body_segment_sensor_lookup[sensor_type][segment_name] = sensor_obj

    def _add_cardinal_sensors(self):
        self.thorax_orient_sensor_lookup = defaultdict(dict)
        for axis_name in ["x", "y", "z"]:
            self.thorax_orient_sensor_lookup[axis_name] = self.mjcf_root.sensor.add(
                f"frame{axis_name}axis",
                name=f"thorax_orient{axis_name}",
                objtype="body", # need to figure out why the order is y z x in the output
                objname="c_thorax",
            )
    

class SingleFlySimulationForRendering(Simulation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.world.fly_lookup) == 1, "Expecting exactly one fly in the simulation."
        assert len(self.world.fly_lookup["nmf"].cameraname_to_mjcfcamera)
        self._map_internal_meshstatesensor_ids()
        self._map_internal_cardinalsensor_ids()
    
    def _map_internal_meshstatesensor_ids(self) -> None:
        internal_meshstatesensorids_by_fly = defaultdict(list)
        for fly_name, fly in self.world.fly_lookup.items():
            if len(fly.body_segment_sensor_lookup) == 0:
                continue
            for segment_name in fly.all_body_segments_names:
                sensor = fly.body_segment_sensor_lookup["pos_atparent"][segment_name]
                internal_id = mj.mj_name2id(
                    self.mj_model, mj.mjtObj.mjOBJ_SENSOR, sensor.full_identifier
                )
                start_idx = self.mj_model.sensor_adr[internal_id]
                sensor_dim = self.mj_model.sensor_dim[internal_id]
                internal_meshstatesensorids_by_fly[fly_name].extend(
                    np.arange(start_idx, start_idx + sensor_dim, dtype=np.int32)
                    )

        
        self._internal_meshstatesensorids_by_fly = {
            k: np.array(v, dtype=np.int32) for k, v in internal_meshstatesensorids_by_fly.items()
        }

    def get_mesh_state_info(self, fly_name:str)-> tuple[
        Float[np.ndarray, "n_segments(69) 3"],  # framepos xyz for the 69 segments
    ]:
        internal_meshstatesensorids = self._internal_meshstatesensorids_by_fly[fly_name]
        sensor_data = self.mj_data.sensordata[internal_meshstatesensorids]
        return sensor_data.reshape(-1, 3)
    
    def _map_internal_cardinalsensor_ids(self) -> None:
        internal_cardinalsensorids_by_fly = defaultdict(dict)
        for fly_name, fly in self.world.fly_lookup.items():
            for axis_name in ["x", "y", "z"]:
                sensor = fly.thorax_orient_sensor_lookup[axis_name]
                internal_id = mj.mj_name2id(
                    self.mj_model, mj.mjtObj.mjOBJ_SENSOR, sensor.full_identifier
                )
                start_idx = self.mj_model.sensor_adr[internal_id]
                sensor_dim = self.mj_model.sensor_dim[internal_id]
                internal_cardinalsensorids_by_fly[fly_name][axis_name] = np.arange(start_idx, start_idx + sensor_dim, dtype=np.int32)
        self._internal_cardinalsensorids_by_fly = internal_cardinalsensorids_by_fly

    def get_cardinal_vectors(self, fly_name:str) -> Float[np.ndarray, "3 3"]:
        internal_cardinalsensorids = self._internal_cardinalsensorids_by_fly[fly_name]
        x_axis = self.mj_data.sensordata[internal_cardinalsensorids["x"]].copy()
        y_axis = self.mj_data.sensordata[internal_cardinalsensorids["y"]].copy()
        z_axis = self.mj_data.sensordata[internal_cardinalsensorids["z"]].copy()
        # assert orthonormality of the axes
        assert np.isclose(x_axis @ y_axis, 0, atol=1e-3), f"x and y axes are not orthogonal: dot product is {x_axis @ y_axis}"
        assert np.isclose(x_axis @ z_axis, 0, atol=1e-3), f"x and z axes are not orthogonal: dot product is {x_axis @ z_axis}"
        return np.stack([x_axis, y_axis, z_axis], axis=0)

def set_up_simulation(render_window_size, render_play_speed, render_fps, sim_timestep, visual_paths):

    # This controlls how strongly the actuators try to track the target joint angles
    actuator_gain = 150.0  # in uN*mm/rad (torque applied per angular discrepancy)
    
    fly = FlyForRendering(visual_paths=visual_paths)
    fly.add_joints(skeleton, neutral_pose=neutral_pose)
    
    actuated_dofs_list = fly.skeleton.get_actuated_dofs_from_preset(actuated_dofs)
    fly.add_actuators(
        actuated_dofs_list,
        actuator_type=actuator_type,
        kp=actuator_gain,
        neutral_input=neutral_pose,
    )
    
    fly.mjcf_root.visual.get_children("global").offwidth = render_window_size[0]
    fly.mjcf_root.visual.get_children("global").offheight = render_window_size[1]

    fly.mjcf_root.root.option.timestep = sim_timestep

    assert fly.name == "nmf", "Expecting fly name to be 'nmf' for consistency with sensors and data recording."

    fly.colorize(0)
    cam_offset = (0, 0, -100)
    cam_rot = Rotation3D(format="euler", values=(0, np.pi, -np.pi / 2))
    cam_offset = (0, 10, 0)
    cam_rot = Rotation3D(format="euler", values=(0, np.pi/2, 0))
    tracking_cam = fly.add_tracking_camera()#pos_offset=cam_offset, rotation=cam_rot, fovy=5)
    fly.add_leg_adhesion()

    spawn_pos = [0, 0, 0.7]  # center of thorax is at 0.7 mm above the ground
    spawn_rot = Rotation3D(format="quat", values=[1, 0, 0, 0])  # no rotation

    world = SpotlightArena()
    world.add_fly(fly, spawn_pos, spawn_rot)

    sim = SingleFlySimulationForRendering(world)

    # The FlyGym camera receives two parameters: a FPS and a play speed. For example,
    # you might tell the camera "I want a video that replays my simulation at 0.1 speed,
    # and I want this slow-mo video to have a FPS of 25 FPS." You give these parameters
    # to the camera, and the camera will figure out "I need save a snapshot every
    # play_speed/fps = 0.1/25 = 0.004 seconds."
    # However, in our case, we want to have a fixed data saving/rendering speed, for
    # example 300 Hz to match Spotlight recordings. We might also want a nice number
    # (e.g. 0.1x) for the play speed. This means that instead of deciding the effective
    # data saving/rendering rate based on the desired play speed and FPS, we need to
    # decide the FPS based on the desired play speed and the target data rate.
    flygym_camera_fps = render_fps * render_play_speed
    renderers = [
        Renderer(
            sim.mj_model,
            tracking_cam,
            camera_res=render_window_size,
            playback_speed=render_play_speed,
            output_fps=flygym_camera_fps,
        )
        for _ in range(len(visual_paths)) # one renderer per visual path/color coding 
    ]

    # Set up state sensors for body segments
    # body_segment_sensor_lookup = {}
    # for sensor_type, sensors_dict in fly.body_segment_sensor_lookup.items():
    #     body_segment_sensor_lookup[sensor_type] = {
    #         "segments_list": list(sensors_dict.keys()),
    #         "sensor_names_list": [sensor.name for sensor in sensors_dict.values()],
    #         "sensor_mjcf_objects_list": list(sensors_dict.values()),
    #         "bound_sensors_list": sim.physics.bind(sensors_dict.values()),
    #     }

    return sim, renderers, tracking_cam


def run_neuromechfly_simulation(
    trajectories_interp,
    render_window_size,
    render_play_speed,
    render_fps,
    sim_timestep,
    min_sim_duration_sec,
    visual_paths,
    max_sim_steps=None,
):
    sim, renderers, cam = set_up_simulation(
        render_window_size, render_play_speed, render_fps, sim_timestep, visual_paths
    )

    # list[float]
    timestamps_hist = []
    # list[ndarray of shape (n_joints,)]
    joints_angles_hist = []
    body_segment_state_hists = {
        # Determined through framepos and framequat sensors added to each body segment:
        "pos_atparent": [],  # list[ndarray of shape (n_segments, 3)]
        "pos_com": [],  # list[ndarray of shape (3,)]
        "quat_atparent": [],  # list[ndarray of shape (n_segments, 4)]
        "quat_com": [],  # list[ndarray of shape (4,)]
        # Global xpos/xquat accessed through MjData:
        "pos_global": [],
        "quat_global": [],
    }
    # list[ndarray of shape (3, 3), ie. (forward/left/up, x/y/z)]
    cardinal_vectors_hist = []
    # list[ndarray of shape (3, 4)], see https://en.wikipedia.org/wiki/Camera_matrix
    camera_matrix_hist = []
    # list[ndarray of shape (3,)], fly base position in world coordinates
    fly_base_pos_hist = []

    all_bodies = sim.world.fly_lookup["nmf"].all_body_segments_names
    base_pose_idx = all_bodies.index("c_thorax")  # used to track fly base position and flipping

    sim.warmup()
    # Simulation loop
    for sim_frame_id in trange(trajectories_interp.shape[0], disable=None):
        # Stop if we have reached the desired number of simulation steps (for testing)
        if max_sim_steps is not None and sim_frame_id >= max_sim_steps:
            break

        # Step physics simulation
        sim.set_actuator_inputs("nmf",
                                actuator_type,
                                trajectories_interp[sim_frame_id])
        sim.step()

        curr_time = sim.time
        if curr_time >= renderers[0]._last_render_time_sec + renderers[0]._secs_between_renders:
            for i, renderer in enumerate(renderers):
                sim.world.fly_lookup["nmf"].colorize(i)  # update fly's visual appearance to match the renderer's visual path/color coding
                ret = renderer.render_as_needed(sim.mj_data)
                assert ret, "Expect rendering to be needed but render_as_needed() returned False."
        else:
            continue  # skip rendering and data recording if there is no frame

        # Store timestamps
        timestamps_hist.append(sim.time)

        # Store joint angles
        joints_angles_hist.append(sim.get_joint_angles("nmf"))

        # Store mesh states (from sensors added to each body segment)
        state_sensor_data = sim.get_mesh_state_info("nmf")
        body_segment_state_hists["pos_atparent"].append(state_sensor_data)
        # for sensor_type, sensor_info in body_segment_sensor_lookup.items():
        #     num_sensors = len(sensor_info["segments_list"])
        #     bound_sensors = sensor_info["bound_sensors_list"]
        #     sensor_readings = bound_sensors.sensordata.copy()
        #     sensor_readings = sensor_readings.reshape((num_sensors, -1))
        #     body_segment_state_hists[sensor_type].append(sensor_readings)

        # Store global body segment states (from MjData)
        pos_global = sim.get_body_positions("nmf")
        quat_global = sim.get_body_rotations("nmf")
        body_segment_state_hists["pos_global"].append(pos_global)
        body_segment_state_hists["quat_global"].append(quat_global)

        # Store forward vector
        cardinal_vectors = sim.get_cardinal_vectors("nmf")
        cardinal_vectors_hist.append(cardinal_vectors)  # forward/left/up vectors in world coordinates

        # Store camera matrix
        cam_xmat = renderers[0].get_xmat_for_camera(cam, sim.mj_data)  # assuming all renderers use the same camera
        camera_matrix_hist.append(cam_xmat.reshape(-1))

        # Store fly base position
        fly_base_pos_hist.append(pos_global[base_pose_idx])

        # Check if the fly has flipped over
        up_vector_z_component = cardinal_vectors[1, 2]
        if up_vector_z_component < 0:
            print(f"Fly flipped over at frame {sim_frame_id}. Stopping simulation.")
            break

    final_simulated_time_sec = sim.time

    if final_simulated_time_sec < min_sim_duration_sec or sim_frame_id == 0:
        print(
            f"Simulation failed too early at {final_simulated_time_sec:.3f} sec. "
            f"Discarding results."
        )
        return renderers, None

    hist_dict = {
        "values": {
            "timestamp": timestamps_hist,
            "joint_angles": joints_angles_hist,
            "body_seg_states": body_segment_state_hists,
            "cardinal_vectors": cardinal_vectors_hist,
            "camera_matrix": camera_matrix_hist,
            "fly_base_pos": fly_base_pos_hist,
        },
        "keys": {
            "joint_angles": [j.name for j in sim.world.fly_lookup["nmf"].get_jointdofs_order()],
            "body_segments": sim.world.fly_lookup["nmf"].all_body_segments_names,
            "cardinal_vectors": ["forward", "left", "up"],  # see flygym docs
        },
    }
    return renderers, hist_dict


def make_simulation_data_h5(hist_dict, h5_path: Path):
    """
    Create HDF5 file with simulation data in the same format as
    convert_pickled_dataframes_to_h5.py
    """
    with h5py.File(h5_path, "w") as h5_file:
        n_timesteps = len(hist_dict["values"]["timestamp"])
        h5_file.attrs["n_timesteps"] = n_timesteps

        # Time
        time_ds = h5_file.create_dataset(
            "sim_time",
            data=np.array(hist_dict["values"]["timestamp"], dtype="float32"),
            dtype="float32",
        )
        time_ds.attrs["units"] = "s"
        time_ds.attrs["description"] = "Time in the NeuroMechFly simulation"

        # DoF angles
        joint_angles_arr = np.array(
            hist_dict["values"]["joint_angles"], dtype="float32"
        )
        dof_keys = []
        for dof_name in hist_dict["keys"]["joint_angles"]:
            kchain_name, link_name = parse_nmf_joint_name(dof_name)
            dof_keys.append(f"{kchain_name}{link_name}")

        dof_angles_ds = h5_file.create_dataset(
            "dof_angles", data=joint_angles_arr, dtype="float32"
        )
        dof_angles_ds.attrs["keys"] = dof_keys
        dof_angles_ds.attrs["units"] = "radians"
        dof_angles_ds.attrs["description"] = (
            "Angles of DoFs tracked in the simulation. "
            "This dataset has shape (n_timesteps, n_dofs). The order of the DoFs is "
            "given in the 'keys' attribute."
        )

        # Body segment states
        body_seg_group = h5_file.create_group("body_segment_states")
        segments_order = hist_dict["keys"]["body_segments"]

        # Process each sensor type in the hist_dict
        for sensor_type, sensor_data_list in hist_dict["values"][
            "body_seg_states"
        ].items():
            if not sensor_data_list:  # Skip empty sensor data
                continue

            sensor_data_arr = np.array(sensor_data_list, dtype="float32")
            if sensor_data_arr.size == 0:
                continue

            # Determine the reference frame and data type from sensor_type
            assert sensor_type.count("_") == 1, f"Unexpected sensor type: {sensor_type}"
            valid_ref_frames = ["atparent", "com", "global"]
            pos_or_quat, ref_frame = sensor_type.split("_")
            assert (
                pos_or_quat in ["pos", "quat"] and ref_frame in valid_ref_frames
            ), f"Unexpected sensor type: {sensor_type}"

            if pos_or_quat == "pos":
                keys = ["x", "y", "z"]
                units = "mm"
            else:  # quat
                keys = ["w", "x", "y", "z"]
                units = "quaternion"
            _pos_or_quat_desc = {
                "pos": "Position",
                "quat": "Orientation (as a quaternion)",
            }
            description = (
                f"{_pos_or_quat_desc[pos_or_quat]} of each body segment in the "
                f'"{ref_frame}" reference frame. This dataset has shape '
                f"(n_timesteps, n_segments, {len(keys)}). The order of the segments "
                "is given in the 'keys' attribute."
            )

            # Create the dataset
            this_ds = body_seg_group.create_dataset(
                sensor_type, data=sensor_data_arr, dtype="float32"
            )
            this_ds.attrs["keys"] = keys
            this_ds.attrs["units"] = units
            this_ds.attrs["description"] = description

        # Set body segment group attributes
        body_seg_group.attrs["keys"] = segments_order
        body_seg_group.attrs["description"] = (
            "Position (in mm) and orientation (as quaternions) of each body segment "
            "tracked in the simulation. Values are provided in several reference "
            "frames: 'atparent' corresponds to 'xbody' in MuJoCo: 'the regular frame "
            "of the body (usually centered at the joint with the parent body)'; "
            "`com` corresponds to 'body' in MuJoCo: 'the inertial frame of the body'; "
            "`global` is the position and orientation in the global/world reference "
            "frame. See "
            "https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-framepos "
            "for the distinction between these `body` and `xbody`."
        )

        # Cardinal vectors
        cardinal_vec_group = h5_file.create_group("cardinal_vectors")
        cardinal_vec_group.attrs["keys"] = ["forward", "left", "up"]
        cardinal_vec_group.attrs["description"] = (
            "Unit vectors pointing in cardinal directions from the perspective of the "
            "fly, i.e. vector pointing forward, to the left, and up from the fly's "
            "body. These vectors are in global coordinates and each of them is of "
            "shape (n_timesteps, 3) where 3 are the x/y/z components."
        )

        cardinal_vectors_arr = np.array(
            hist_dict["values"]["cardinal_vectors"], dtype="float32"
        )
        for i, vec_direction in enumerate(["forward", "left", "up"]):
            data_block = cardinal_vectors_arr[:, i, :]  # (n_timesteps, 3)
            this_ds = cardinal_vec_group.create_dataset(
                vec_direction, data=data_block, dtype="float32"
            )
            this_ds.attrs["keys"] = ["x", "y", "z"]

        # Camera matrix
        camera_matrix_arr = np.array(
            hist_dict["values"]["camera_matrix"], dtype="float32"
        )
        cam_mat_ds = h5_file.create_dataset(
            "camera_matrix", data=camera_matrix_arr, dtype="float32"
        )
        cam_mat_ds.attrs["description"] = (
            "3x4 camera matrix (numpy array) for each frame "
            "(see https://en.wikipedia.org/wiki/Camera_matrix)."
        )

        # Fly base position
        fly_base_pos_arr = np.array(
            hist_dict["values"]["fly_base_pos"], dtype="float32"
        )
        fly_base_pos_ds = h5_file.create_dataset(
            "fly_base_pos", data=fly_base_pos_arr, dtype="float32"
        )
        fly_base_pos_ds.attrs["keys"] = ["x", "y", "z"]
        fly_base_pos_ds.attrs["units"] = "mm"
        fly_base_pos_ds.attrs["description"] = (
            "Position of the fly's center of mass in global coordinates. This dataset "
            "has shape (n_timesteps, 3) where the 3 values are the x/y/z components."
        )


def simulate_one_segment(
    kinematic_recording_segment: pd.DataFrame,
    output_dir: Path,
    input_timestep: float,
    sim_timestep: float,
    visual_paths: list[Path],
    output_data_freq: int = 300,
    render_play_speed: float = 0.1,
    render_window_size=(720, 720),
    min_sim_duration_sec: float = 0.2,
    max_sim_steps: int | None = None,
) -> bool:
    """Simulate a single segment of kinematic recording in FlyGym. Note
    that no result will be saved if simulation fails before
    `min_sim_duration_sec` is reached.

    Args:
        kinematic_recording_segment (pd.DataFrame): A segment of kinematic
            recording, as returned by `load_kinematic_recording()`.
        output_dir (Path): Directory to save simulation results.
        input_timestep (float): Timestep of input kinematics (from Aymanns
            et al. 2022).
        sim_timestep (float): Timestep to use in the physics simulation.
        output_data_freq (int): frequency (in Hz) to save data and render
            frames. Note that is not the frequency of the physics. It only
            affects the data saving rate.
        render_play_speed (float): Play speed to use when rendering the
            simulation. This affects neither the simulation itself nor the
            rendering and saving of data during the simulation. It only
            affects how the rendered video is played (i.e. the metadata of
            the output video used by media players).
        render_window_size (tuple[int, int]): Window size to use when
            rendering the simulation.
        min_sim_duration_sec (float): Minimum simulation duration to
            consider the simulation successful.
        max_sim_steps (int | None): If not None, limit the number of
            simulation steps to this number. This is mainly for testing.

    Returns:
        bool: Whether the simulation was successful (i.e. reached
            `min_sim_duration_sec`).

    Example:
        Assume we want to simulate data from Aymanns et al. 2022 in
        NeuroMechFly. We need to understand/decide:

            1. At what frequency were the original kinematics from Aymanns
               et al. recorded? The answer is 100 Hz.
            2. At what frequency do we want to run the physics simulation?
               For stability, let's use a small timestep of 0.0001 sec.
            3. At what frequency do we want to save the simulation data and
               rendered frames? Let's say we want to match the simulated
               video with Spotlight behavior recordings recorded at 300 Hz.
            4. At what speed do we want to play the rendered video? This is
               for visualization only and does not affect the simulation or
               data saving. Let's say we want to use a slow play speed of
               0.1x to better see the details of the fly's movements.

        Then, we should set `output_data_freq` to 300, `input_timestep` to
        0.01, `sim_timestep` to 0.0001, and `render_play_speed` to 0.1 when
        calling this function.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Interpolate the trajectories to match the simulation timestep
    actuated_joints_list = skeleton.get_actuated_dofs_from_preset(actuated_dofs)
    trajectories_interp, interp_factor = interpolate_trajectories(
        kinematic_recording_segment, input_timestep, sim_timestep, actuated_joints_list
    )
    print(f"Simulating segment with {len(kinematic_recording_segment)} frames.")
    print(f"Interpolation factor: {interp_factor}")
    print(f"Output directory: {output_dir}")

    # Run the simulation
    renderers, hist_dict = run_neuromechfly_simulation(
        trajectories_interp,
        render_window_size,
        render_play_speed,
        output_data_freq,
        sim_timestep,
        min_sim_duration_sec,
        visual_paths,
        max_sim_steps=max_sim_steps,
    )

    # Do nothing if simulation failed before the minimum required duration is reached
    if hist_dict is None:
        # Save rendered frames as a video
        for i, renderer in enumerate(renderers):
            video_path = output_dir / f"rendered_video_{visual_paths[i].name}.mp4"
            renderer.save_video(video_path)
        return False

    # Save kinematic states as Pandas DataFrame
    h5_path = output_dir / "simulation_data.h5"
    make_simulation_data_h5(hist_dict, h5_path)

    # Save rendered frames as a video
    for i, renderer in enumerate(renderers):
        video_path = output_dir / f"rendered_video_{visual_paths[i].name}.mp4"
        renderer.save_video(video_path)

    return True
