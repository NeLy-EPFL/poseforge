import numpy as np
import pandas as pd
import dm_control.mujoco
import imageio
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
    fly = FlyForRendering(
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
        last_num_frames = len(camera._frames_by_color_coding[0])

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
    camera.save_video(output_dir)
