import numpy as np
import pandas as pd
import dm_control.mujoco
import imageio
from collections import defaultdict
from tqdm import trange
from pathlib import Path
from dm_control.rl.control import PhysicsError
from dm_control.mjcf import Physics
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
        self._add_mesh_state_sensors()
        self._assign_colors()
        self.current_color_coding: int = -1

    def get_color_combo_rgba(self, segment_name: str):
        return self._segment_name_to_color_combo_rgba.get(
            segment_name, self._default_color_combo_rgba
        )

    def _assign_colors(self):
        palette = {
            "red": (1, 0, 0, 1),
            "green": (0, 1, 0, 1),
            "blue": (0, 0, 1, 1),
            "yellow": (1, 1, 0, 1),
            "magenta": (1, 0, 1, 1),
            "cyan": (0, 1, 1, 1),
            "gray": (0.4, 0.4, 0.4, 1),
            "white": (1, 1, 1, 1),
        }

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
        self._segment_name_to_color_combo_rgba = {}

        # Leg segments
        for side in "LR":
            for pos in "FMH":
                leg = f"{side}{pos}"
                for link in ["Coxa", "Femur", "Tibia"]:
                    segment_name = f"{leg}{link}"
                    color_combo = (
                        palette[color_by_link[link]],
                        palette[color_by_kinematic_chain[leg]],
                    )
                    self._segment_name_to_color_combo_rgba[segment_name] = color_combo

                # All tarsal segments share the same color
                for tarsus_idx in range(1, 6):
                    segment_name = f"{leg}Tarsus{tarsus_idx}"
                    color_combo = (
                        palette[color_by_link["Tarsus"]],
                        palette[color_by_kinematic_chain[leg]],
                    )
                    self._segment_name_to_color_combo_rgba[segment_name] = color_combo

        # Antennal segments
        for side in "LR":
            for link in ["Pedicel", "Funiculus", "Arista"]:
                segment_name = f"{side}{link}"
                color_combo = (
                    palette[color_by_link["Antenna"]],
                    palette[color_by_kinematic_chain[side]],
                )
                self._segment_name_to_color_combo_rgba[segment_name] = color_combo

        # Special case for Thorax
        thorax_color_combo = (
            palette[color_by_link["Thorax"]],
            palette[color_by_kinematic_chain["Thorax"]],
        )
        self._segment_name_to_color_combo_rgba["Thorax"] = thorax_color_combo

        # Set default color combo for segments not listed above
        self._default_color_combo_rgba = (palette["gray"], palette["gray"])

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

        # Rotation in body frame (xbody), i.e. at joint with parent body
        sensors["quat_atparent"] = self.model.sensor.add(
            "framequat",
            name=f"{segment_name}_quat_atparent",
            objtype="xbody",
            objname=segment_name,
        )

        # Position in inertial frame (body), i.e. at COM of body
        sensors["pos_com"] = self.model.sensor.add(
            "framepos",
            name=f"{segment_name}_pos_com",
            objtype="body",
            objname=segment_name,
        )

        # Rotation in inertial frame (body), i.e. at COM of body
        sensors["quat_com"] = self.model.sensor.add(
            "framequat",
            name=f"{segment_name}_quat_com",
            objtype="body",
            objname=segment_name,
        )

        return sensors

    def _add_mesh_state_sensors(self):
        self.body_segment_sensor_lookup = defaultdict(dict)

        # Add sensors to leg segments
        legs = [f"{side}{pos}" for side in "LR" for pos in "FMH"]
        leg_links = ["Coxa", "Femur", "Tibia"] + [f"Tarsus{i}" for i in range(1, 6)]
        for leg in legs:
            # Add keypoint position sensors for each link except the claws
            # (which are already handled by FlyGym's end effector sensors)
            for link in leg_links:
                seg_name = f"{leg}{link}"
                sensors = self._add_pos_and_quat_sensors(seg_name)
                for sensor_type, sensor_obj in sensors.items():
                    self.body_segment_sensor_lookup[sensor_type][seg_name] = sensor_obj

        # Add antennal segments
        sides = ["L", "R"]
        antennal_links = ["Pedicel", "Funiculus", "Arista"]
        for side in sides:
            for link in antennal_links:
                seg_name = f"{side}{link}"
                sensors = self._add_pos_and_quat_sensors(seg_name)
                for sensor_type, sensor_obj in sensors.items():
                    self.body_segment_sensor_lookup[sensor_type][seg_name] = sensor_obj

        # Add thorax
        seg_name = "Thorax"
        sensors = self._add_pos_and_quat_sensors(seg_name)
        for sensor_type, sensor_obj in sensors.items():
            self.body_segment_sensor_lookup[sensor_type][seg_name] = sensor_obj


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
       put the rendered frame into the appropriate list in `self._frames_by_color_coding`.
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
    fly = FlyForRendering(
        init_pose="stretch",
        control="position",
        actuated_joints=all_leg_dofs,
        xml_variant="deepfly3d",
        spawn_pos=(0, 0, 0.5),
        joint_stiffness=0.1,
    )
    camera_params = {
        "mode": "track",
        "pos": (0, 0, -100),
        "euler": (0, np.pi, -np.pi / 2),
        "fovy": 5,
    }
    camera = CameraForRendering(
        attachment_point=fly.model.worldbody,
        camera_name=f"flytrack_cam",
        camera_parameters=camera_params,
        window_size=render_window_size,
        play_speed=render_play_speed,
        fps=render_fps * render_play_speed,
        draw_contacts=False,
        play_speed_text=False,
    )
    sim = SingleFlySimulationForRendering(
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
    trajectories_interp, render_window_size, render_play_speed, render_fps, sim_timestep
):
    sim, camera, dm_camera, body_segment_sensor_lookup = set_up_simulation(
        render_window_size, render_play_speed, render_fps, sim_timestep
    )

    timestamps_hist = []  # list[float]
    joints_angles_hist = []  # list[ndarray of shape (n_joints,)]
    body_segment_state_hists = {
        "pos_atparent": [],  # list[ndarray of shape (n_segments, 3)]
        "pos_com": [],  # list[ndarray of shape (3,)]
        "quat_atparent": [],  # list[ndarray of shape (n_segments, 4)]
        "quat_com": [],  # list[ndarray of shape (4,)]
    }
    cardinal_vectors_hist = []  # list[ndarray of shape (3, 3), ie (fwd/left/up, x/y/z)]
    camera_matrix_hist = (
        []
    )  # list[ndarray of shape (3, 4)]see https://en.wikipedia.org/wiki/Camera_matrix

    # Simulation loop
    for sim_frame_id in trange(trajectories_interp.shape[0], disable=None):
        # if sim_frame_id == 1000: break  # For debugging, remove later
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

        # Store mesh states
        for sensor_type, sensor_info in body_segment_sensor_lookup.items():
            num_sensors = len(sensor_info["segments_list"])
            bound_sensors = sensor_info["bound_sensors_list"]
            sensor_readings = bound_sensors.sensordata.copy().reshape((num_sensors, -1))
            body_segment_state_hists[sensor_type].append(sensor_readings)

        # Store forward vector
        cardinal_vectors_hist.append(observation["cardinal_vectors"].copy())

        # Store camera matrix
        camera_matrix_hist.append(dm_camera.matrix.copy())

        # Check if the fly has flipped over
        if observation["cardinal_vectors"][2, 2] < 0:
            print(f"Fly flipped over at frame {sim_frame_id}. Stopping simulation.")
            break

    hist_dict = {
        "values": {
            "timestamp": timestamps_hist,
            "joint_angles": joints_angles_hist,
            "body_seg_states": body_segment_state_hists,
            "cardinal_vectors": cardinal_vectors_hist,
            "camera_matrix": camera_matrix_hist,
        },
        "keys": {
            "joint_angles": sim.fly.actuated_joints,
            # body_seg_states: all sensor types share the same list of body segments.
            # Use pos_com just as an example
            "body_seg_states": body_segment_sensor_lookup["pos_com"]["segments_list"],
            "cardinal_vectors": ["forward", "left", "up"],  # see flygym docs
        },
    }
    return camera, hist_dict


def make_kinematic_states_dataframe(hist_dict):
    columns = {}
    columns["time"] = hist_dict["values"]["timestamp"]

    # Joint angles
    joint_angles_arr = np.array(hist_dict["values"]["joint_angles"], dtype=np.float32)
    for i, dof_name in enumerate(hist_dict["keys"]["joint_angles"]):
        kchain_name, link_name = parse_nmf_joint_name(dof_name)
        columns[f"dof_angle_{kchain_name}{link_name}"] = joint_angles_arr[:, i]

    # Body segment states
    for sensor_type, sensor_data in hist_dict["values"]["body_seg_states"].items():
        # sensor_data_arr has shape shape (nframes, nsegs, 3 for pos or 4 quaternion)
        sensor_data_arr = np.array(sensor_data, dtype=np.float32)
        for i, seg_name in enumerate(hist_dict["keys"]["body_seg_states"]):
            list_of_arrs = list(sensor_data_arr[:, i, :])
            columns[f"body_seg_{sensor_type}_{seg_name}"] = list_of_arrs

    # Cardinal vectors
    cardinal_vectors_arr = np.array(
        hist_dict["values"]["cardinal_vectors"], dtype=np.float32
    )
    for i, direction in enumerate(hist_dict["keys"]["cardinal_vectors"]):
        columns[f"cardinal_vector_{direction}"] = list(cardinal_vectors_arr[:, i, :])

    # Camera matrix
    columns["camera_matrix"] = [
        x.astype(np.float32) for x in hist_dict["values"]["camera_matrix"]
    ]

    kinematic_states_df = pd.DataFrame(columns)
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
    camera, hist_dict = run_neuromechfly_simulation(
        trajectories_interp,
        render_window_size,
        render_play_speed,
        render_fps,
        sim_timestep,
    )

    # Save kinematic states as Pandas DataFrame
    kinematic_states_df = make_kinematic_states_dataframe(hist_dict)
    kinematic_states_df.to_pickle(output_dir / "kinematic_states_history.pkl")

    # Save rendered frames as a video
    camera.save_video(output_dir)
