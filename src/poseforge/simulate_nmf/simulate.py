import numpy as np
import pandas as pd
import dm_control.mujoco
import imageio
import h5py
from collections import defaultdict
from tqdm import trange
from pathlib import Path
from dm_control.rl.control import PhysicsError
from dm_control.mjcf import Physics
from flygym import Fly, SingleFlySimulation, Camera
from flygym.arena import BaseArena, FlatTerrain
from flygym.preprogrammed import all_leg_dofs

from poseforge.simulate_nmf.data import interpolate_trajectories
from poseforge.simulate_nmf.constants import parse_nmf_joint_name


# Define color combo by body segment
color_by_link = {
    "Coxa": "cyan",
    "Femur": "yellow",
    "Tibia": "blue",
    "Tarsus": "green",
    "Antenna": "magenta",
    "Thorax": "gray",
}
color_by_kinematic_chain = {
    "LF": "red",  # left front leg
    "LM": "green",  # left mid leg
    "LH": "blue",  # left hind leg
    "RF": "cyan",  # right front leg
    "RM": "magenta",  # right mid leg
    "RH": "yellow",  # right hind leg
    "L": "red",  # left antenna
    "R": "green",  # right antenna
    "Thorax": "white",  # thorax
}
color_palette = {
    "red": (1, 0, 0, 1),
    "green": (0, 1, 0, 1),
    "blue": (0, 0, 1, 1),
    "yellow": (1, 1, 0, 1),
    "magenta": (1, 0, 1, 1),
    "cyan": (0, 1, 1, 1),
    "gray": (0.4, 0.4, 0.4, 1),
    "white": (1, 1, 1, 1),
}


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

        # Remove lights
        for light in self.root_element.root.worldbody.light:
            light.remove()

        # Make lights non-shadow-casting to make segments look more uniform
        self.root_element.default.light.castshadow = False

    def get_spawn_position(self, rel_pos, rel_angle):
        return rel_pos, rel_angle


class FlyForRendering(Fly):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_body_segment_names = [
            body.name for body in self.model.find_all("body") if body.name != "FlyBody"
        ]
        self._add_mesh_state_sensors()
        self._assign_colors()
        self.current_color_coding: int = -1

    def get_color_combo_rgba(self, segment_name: str):
        return self._segment_name_to_color_combo_rgba.get(
            segment_name, self._default_color_combo_rgba
        )

    def _assign_colors(self):
        self._segment_name_to_color_combo_rgba = {}

        # Leg segments
        for side in "LR":
            for pos in "FMH":
                leg = f"{side}{pos}"
                for link in ["Coxa", "Femur", "Tibia"]:
                    segment_name = f"{leg}{link}"
                    color_combo = (
                        color_palette[color_by_link[link]],
                        color_palette[color_by_kinematic_chain[leg]],
                    )
                    self._segment_name_to_color_combo_rgba[segment_name] = color_combo

                # All tarsal segments share the same color
                for tarsus_idx in range(1, 6):
                    segment_name = f"{leg}Tarsus{tarsus_idx}"
                    color_combo = (
                        color_palette[color_by_link["Tarsus"]],
                        color_palette[color_by_kinematic_chain[leg]],
                    )
                    self._segment_name_to_color_combo_rgba[segment_name] = color_combo

        # Antennal segments
        for side in "LR":
            for link in ["Pedicel", "Funiculus", "Arista"]:
                segment_name = f"{side}{link}"
                color_combo = (
                    color_palette[color_by_link["Antenna"]],
                    color_palette[color_by_kinematic_chain[side]],
                )
                self._segment_name_to_color_combo_rgba[segment_name] = color_combo

        # Special case for Thorax
        thorax_color_combo = (
            color_palette[color_by_link["Thorax"]],
            color_palette[color_by_kinematic_chain["Thorax"]],
        )
        self._segment_name_to_color_combo_rgba["Thorax"] = thorax_color_combo

        # Set default color combo for segments not listed above
        self._default_color_combo_rgba = (color_palette["gray"], color_palette["gray"])

    def _set_geom_colors(self):
        """This method is inherited fly flygym.Fly, but we override it to
        set material properties. We don't really set colors (they are
        handled upon rendering). We simply set the material property to
        make the label rendering look uniform.
        """
        # Define material for each different color to be rendered
        self.model.asset.add(
            "material",
            name=f"seglabel_material",
            emission=1.0,
            specular=0.0,
            shininess=0.0,
            reflectance=0.0,
        )

        # Assign material to each geom
        for geom in self.model.find_all("geom"):
            if hasattr(geom, "material"):
                geom.material = f"seglabel_material"
            else:
                print(f"Geom {geom.name} has no material attribute")
            geom._remove_attribute("rgba")

    def change_color_coding(self, physics: Physics, color_coding_idx: int):
        if color_coding_idx == self.current_color_coding:
            return
        for geom in self.model.find_all("geom"):
            rgba = self.get_color_combo_rgba(geom.name)[color_coding_idx]
            self.change_segment_color(physics, geom.name, rgba)

    def _add_pos_and_quat_sensors(self, segment_name):
        sensors = {}

        # Position in body frame (xbody), i.e. at joint with parent body
        sensors["pos_atparent"] = self.model.sensor.add(
            "framepos",
            name=f"{segment_name}_pos_atparent",
            objtype="xbody",
            objname=segment_name,
        )

        # The following sensors are not currently used, so we comment them out to
        # avoid cluttering the simulation with too many sensors.
        # # Rotation in body frame (xbody), i.e. at joint with parent body
        # sensors["quat_atparent"] = self.model.sensor.add(
        #     "framequat",
        #     name=f"{segment_name}_quat_atparent",
        #     objtype="xbody",
        #     objname=segment_name,
        # )

        # # Position in inertial frame (body), i.e. at COM of body
        # sensors["pos_com"] = self.model.sensor.add(
        #     "framepos",
        #     name=f"{segment_name}_pos_com",
        #     objtype="body",
        #     objname=segment_name,
        # )

        # # Rotation in inertial frame (body), i.e. at COM of body
        # sensors["quat_com"] = self.model.sensor.add(
        #     "framequat",
        #     name=f"{segment_name}_quat_com",
        #     objtype="body",
        #     objname=segment_name,
        # )

        return sensors

    def _add_mesh_state_sensors(self):
        self.body_segment_sensor_lookup = defaultdict(dict)
        for segment_name in self.all_body_segment_names:
            sensors = self._add_pos_and_quat_sensors(segment_name)
            for sensor_type, sensor_obj in sensors.items():
                self.body_segment_sensor_lookup[sensor_type][segment_name] = sensor_obj


class SingleFlySimulationForRendering(SingleFlySimulation):
    num_color_codings = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.cameras) == 1, "Only single camera supported."
        assert len(self.flies) == 1, "Only single fly supported."

    def render(self):
        images = []
        for color_coding_idx in range(self.num_color_codings):
            self.fly.change_color_coding(self.physics, color_coding_idx)
            img = self.cameras[0].render(
                self.physics, self._floor_height, self.curr_time, color_coding_idx
            )
            if img is None:
                return None
            images.append(img)
        return images


class CameraForRendering(Camera):
    """Modified from `flygym.Camera.render()` to support multiple color codings.

    The functional differences are:
    1. The `CameraForRendering` class maintains a `self._frames_by_color_coding` list
       containing `num_color_codings` lists of frames, instead of a single
       `self._frames list`.
    2. The `.render()` method receives an additional `color_coding_idx` that is used to
       put the rendered frame into the appropriate list in
       `self._frames_by_color_coding`.
    3. The `.save_video()` method saves a separate video for each color coding.
    """

    num_color_codings = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self._frames
        self._frames_by_color_coding = [[] for _ in range(self.num_color_codings)]

    def render(
        self,
        physics: Physics,
        floor_height: float,
        curr_time: float,
        color_coding_idx: int,
        last_obs: dict | None = None,
    ) -> np.ndarray | None:
        """Mostly copied from `flygym.Camera.render()`"""
        if (
            curr_time
            < len(self._frames_by_color_coding[color_coding_idx])
            * self._eff_render_interval
        ):
            return None

        if last_obs is not None:
            self._update_camera(physics, floor_height, last_obs[0])

        width, height = self.window_size
        img = physics.render(width=width, height=height, camera_id=self.camera_id)
        img = img.copy()
        if last_obs is not None:
            if self.draw_contacts:
                for i in range(len(self.targeted_fly_names)):
                    img = self._draw_contacts(img, physics, last_obs[i])
            if self.draw_gravity:
                img = self._draw_gravity(img, physics, last_obs[0]["pos"])

        self._frames_by_color_coding[color_coding_idx].append(img)
        self._timestamp_per_frame.append(curr_time)
        return img

    def save_video(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)

        for color_coding_idx in range(self.num_color_codings):
            path = save_dir / f"nmf_sim_render_colorcode_{color_coding_idx}.mp4"
            with imageio.get_writer(path, fps=self.fps) as video_writer:
                for img in self._frames_by_color_coding[color_coding_idx]:
                    video_writer.append_data(img)


def set_up_simulation(render_window_size, render_play_speed, render_fps, sim_timestep):
    # Set up fly
    fly = FlyForRendering(
        init_pose="stretch",
        control="position",
        actuated_joints=all_leg_dofs,
        xml_variant="deepfly3d",
        spawn_pos=(0, 0, 0.5),
        joint_stiffness=0.1,
    )

    # Set up camera
    camera_params = {
        "mode": "track",
        "pos": (0, 0, -100),
        "euler": (0, np.pi, -np.pi / 2),
        "fovy": 5,
    }
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
    camera = CameraForRendering(
        attachment_point=fly.model.worldbody,
        camera_name=f"flytrack_cam",
        camera_parameters=camera_params,
        window_size=render_window_size,
        play_speed=render_play_speed,
        fps=flygym_camera_fps,
        draw_contacts=False,
        play_speed_text=False,
    )

    # Set up simulation
    sim = SingleFlySimulationForRendering(
        fly=fly,
        cameras=[camera],
        arena=SpotlightArena(),
        timestep=sim_timestep,
    )
    # Get a dm_control.mujoco.Camera object shadowing the FlyGym Camera
    # that we set up above. This is used to extract the camera matrix more
    # easily during simulation.
    dm_camera = dm_control.mujoco.Camera(
        sim.physics,
        camera_id=camera.camera_id,
        width=camera.window_size[0],
        height=camera.window_size[1],
    )

    # Set up state sensors for body segments
    body_segment_sensor_lookup = {}
    for sensor_type, sensors_dict in fly.body_segment_sensor_lookup.items():
        body_segment_sensor_lookup[sensor_type] = {
            "segments_list": list(sensors_dict.keys()),
            "sensor_names_list": [sensor.name for sensor in sensors_dict.values()],
            "sensor_mjcf_objects_list": list(sensors_dict.values()),
            "bound_sensors_list": sim.physics.bind(sensors_dict.values()),
        }

    return sim, camera, dm_camera, body_segment_sensor_lookup


def run_neuromechfly_simulation(
    trajectories_interp,
    render_window_size,
    render_play_speed,
    render_fps,
    sim_timestep,
    min_sim_duration_sec,
    max_sim_steps=None,
):
    sim, camera, dm_camera, body_segment_sensor_lookup = set_up_simulation(
        render_window_size, render_play_speed, render_fps, sim_timestep
    )
    body_prefix = [
        x for x in sim.physics.named.data.xpos.axes.row.names if x.endswith("/FlyBody")
    ]
    assert (
        len(body_prefix) == 1
    ), r"Expecting exactly one body named '{prefix}/FlyBody'."
    body_prefix = body_prefix[0].split("/")[0]  # should be something like "0/" or "1/"
    body_segments_list_with_prefix = [
        f"{body_prefix}/{segment_name}"
        for segment_name in sim.fly.all_body_segment_names
    ]

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

    # Simulation loop
    for sim_frame_id in trange(trajectories_interp.shape[0], disable=None):
        # Stop if we have reached the desired number of simulation steps (for testing)
        if max_sim_steps is not None and sim_frame_id >= max_sim_steps:
            break

        # Step physics simulation
        action = {"joints": trajectories_interp[sim_frame_id]}
        try:
            observation, reward, terminated, truncated, info = sim.step(action)
        except PhysicsError as e:
            print(f"Simulation error at frame {sim_frame_id}: {e}")
            break
        frame_rendered = sim.render()

        # Check if a new frame is rendered. If not, continue without extracting any data
        if frame_rendered is None:
            continue

        # Store timestamps
        timestamps_hist.append(sim.curr_time)

        # Store joint angles
        joints_angles_hist.append(observation["joints"][0, :].copy())

        # Store mesh states (from sensors added to each body segment)
        for sensor_type, sensor_info in body_segment_sensor_lookup.items():
            num_sensors = len(sensor_info["segments_list"])
            bound_sensors = sensor_info["bound_sensors_list"]
            sensor_readings = bound_sensors.sensordata.copy()
            sensor_readings = sensor_readings.reshape((num_sensors, -1))
            body_segment_state_hists[sensor_type].append(sensor_readings)

        # Store global body segment states (from MjData)
        pos_global = sim.physics.named.data.xpos[body_segments_list_with_prefix]
        quat_global = sim.physics.named.data.xquat[body_segments_list_with_prefix]
        body_segment_state_hists["pos_global"].append(pos_global)
        body_segment_state_hists["quat_global"].append(quat_global)

        # Store forward vector
        cardinal_vectors_hist.append(observation["cardinal_vectors"].copy())

        # Store camera matrix
        camera_matrix_hist.append(dm_camera.matrix.copy())

        # Store fly base position
        fly_base_pos = observation["fly"][0, :].copy()
        fly_base_pos_hist.append(fly_base_pos)

        # Check if the fly has flipped over
        up_vector_z_component = observation["cardinal_vectors"][2, 2]
        if up_vector_z_component < 0:
            print(f"Fly flipped over at frame {sim_frame_id}. Stopping simulation.")
            break

    final_simulated_time_sec = sim.curr_time
    sim.close()

    if final_simulated_time_sec < min_sim_duration_sec or sim_frame_id == 0:
        print(
            f"Simulation failed too early at {final_simulated_time_sec:.3f} sec. "
            f"Discarding results."
        )
        return camera, None

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
            "joint_angles": sim.fly.actuated_joints,
            "body_segments": sim.fly.all_body_segment_names,
            "cardinal_vectors": ["forward", "left", "up"],  # see flygym docs
        },
    }
    return camera, hist_dict


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
    trajectories_interp, interp_factor = interpolate_trajectories(
        kinematic_recording_segment, input_timestep, sim_timestep
    )
    print(f"Simulating segment with {len(kinematic_recording_segment)} frames.")
    print(f"Interpolation factor: {interp_factor}")
    print(f"Output directory: {output_dir}")

    # Run the simulation
    camera, hist_dict = run_neuromechfly_simulation(
        trajectories_interp,
        render_window_size,
        render_play_speed,
        output_data_freq,
        sim_timestep,
        min_sim_duration_sec,
        max_sim_steps=max_sim_steps,
    )

    # Do nothing if simulation failed before the minimum required duration is reached
    if hist_dict is None:
        return False

    # Save kinematic states as Pandas DataFrame
    h5_path = output_dir / "simulation_data.h5"
    make_simulation_data_h5(hist_dict, h5_path)

    # Save rendered frames as a video
    camera.save_video(output_dir)

    return True
