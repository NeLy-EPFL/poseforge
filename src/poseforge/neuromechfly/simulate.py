import numpy as np
import pandas as pd
import imageio
import h5py
from collections import defaultdict
from tqdm import trange
from pathlib import Path
import matplotlib.pyplot as plt

from flygym.compose import Fly, KinematicPose, ActuatorType
from flygym.compose.fly import FlyBody
from flygym.flybody.anatomy_flybody import (
    FlyBodySkeleton,
    FlyBodyJointPreset,
    FlyBodyAxisOrder,
    FlyBodyActuatedDOFPreset,
    FlyBodyContactBodiesPreset,
)
from flygym.anatomy import Skeleton, AxisOrder, JointPreset, ActuatedDOFPreset, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym import Simulation, Renderer

import mujoco as mj
import dm_control.mjcf as mjcf
import numpy as np
from jaxtyping import Float

from poseforge.neuromechfly.data import interpolate_trajectories
from poseforge.neuromechfly.constants import (
    parse_nmf_joint_seg,
)

actuator_type = ActuatorType.POSITION
nofuse_global = Path(__file__).parent / "mujoco_globals_nofuse.yaml"

class SpotlightArena(FlatGroundWorld):
    def __init__(
        self,
        name: str = "spotlight_arena",
        size: tuple[float, float] = (100, 100),
        friction: tuple[float, float, float] = (1, 0.005, 0.0001),
        background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        # Stash this before super().__init__() because BaseWorld.__init__
        # calls self._add_skybox() — which we override below to pick up the
        # requested colour rather than the white flygym default.
        self._background_rgb = tuple(background_rgb)
        super().__init__(name)
        self.ground_geom.rgba = [1, 1, 1, 0]  # make the textured ground transparent # remove the ground with texture add transparent floor

        # Keep the same plane (now invisible); only disable its collision
        # affinity and set friction. MjSpec geoms cannot be removed via an
        # element method, and reusing the plane keeps the world's contact pairs
        # valid.
        self.ground_geom.conaffinity = 0
        self.ground_geom.friction = list(friction)
        # Keep ground_geoms in sync with the replaced ground_geom; otherwise
        # add_fly() builds <pair> contacts pointing at the removed geom,
        # which dm_control surfaces as "Attribute 'geom2' of element <pair>
        # is required" during XML compile.
        self.ground_geoms = [self.ground_geom]
        self.friction = friction

        # # Make lights non-shadow-casting to make segments look more uniform
        # self.root_element.default.light.castshadow = False

        self.legpos_to_groundcontactsensors_by_fly = None

    def get_spawn_position(self, rel_pos, rel_angle):
        return rel_pos, rel_angle

    def _add_skybox(self):
        # Override the flygym default (white gradient skybox) so the rendered
        # background matches the dark experimental Spotlight recordings.
        r, g, b = self._background_rgb
        self.mjcf_root.add_texture(
            name="skybox",
            type=mj.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mj.mjtBuiltin.mjBUILTIN_FLAT,
            rgb1=(r, g, b),
            rgb2=(r, g, b),
            width=10,
            height=10,
        )


class FlyForRendering(Fly):
    def __init__(self, *args, **kwargs):
        self.all_material_names = []
        super().__init__(*args, **kwargs)
        self.sensorized_body_segments = []
        
    def _add_pos_and_quat_sensors(self, body):

        sensors = {}
        b_name = body.name
        self.sensorized_body_segments.append(b_name)

        # Position in body frame (xbody), i.e. at joint with parent body
        sensors["pos_atparent"] = self.mjcf_root.add_sensor(
            name=f"{b_name}_pos_atparent",
            type=mj.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mj.mjtObj.mjOBJ_XBODY,
            objname=b_name,
        )

        # The following sensors are not currently used, so we comment them out to
        # avoid cluttering thxe simulation with too many sensors.
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

    def add_mesh_state_sensors(self):
        self.body_segment_sensor_lookup = defaultdict(dict)
        for body in self.mjcf_root.bodies:
            # Skip the unnamed MjSpec worldbody and the fly spec's top-level
            # "world" attachment body: the latter is merged into the arena on
            # attach, so no sensor can reference it in the compiled model.
            if not body.name or body.name == "world":
                continue
            sensors = self._add_pos_and_quat_sensors(body)
            for sensor_type, sensor_obj in sensors.items():
                self.body_segment_sensor_lookup[sensor_type][body.name] = sensor_obj

    def add_cardinal_sensors(self):
        self.thorax_orient_sensor_lookup = defaultdict(dict)
        axis_to_sensor_type = {
            "x": mj.mjtSensor.mjSENS_FRAMEXAXIS,
            "y": mj.mjtSensor.mjSENS_FRAMEYAXIS,
            "z": mj.mjtSensor.mjSENS_FRAMEZAXIS,
        }
        for axis_name in ["x", "y", "z"]:
            self.thorax_orient_sensor_lookup[axis_name] = self.mjcf_root.add_sensor(
                name=f"thorax_orient{axis_name}",
                type=axis_to_sensor_type[axis_name],
                objtype=mj.mjtObj.mjOBJ_XBODY,  # order is y z x in the output
                objname="c_thorax",
            )
        
    def add_tracking_camera(self, **kwargs):
        # The camera is attached directly to the thorax (the fly root) by flygym; no
        # extra body is added, so the fly's dynamics are unchanged. (A separate mount
        # body cannot be made massless -- the model's compiler boundmass=1e-6 floor
        # gives any body at least 1e-6 kg, which perturbs the replay and flips the fly
        # -- so camera framing must not rely on an added body.)
        cam = super().add_tracking_camera(**kwargs)

        # add a light to the thorax
        # Point the spotlight the same way as the tracking camera. MjSpec
        # cameras expose `pos` but not `euler`, so take the orientation from the
        # Rotation3D passed in (euler format, as used by set_up_simulation).
        from scipy.spatial.transform import Rotation

        pos_offset = cam.pos
        rotation_3d = kwargs.get("rotation")
        if rotation_3d is not None and rotation_3d.format == "euler":
            camera_rotation = Rotation.from_euler("xyz", rotation_3d.values)
        else:
            quat = cam.quat  # MuJoCo quaternion order (w, x, y, z)
            camera_rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        view_direction = camera_rotation.apply([0, 0, -1])
        camera_right = camera_rotation.apply([1, 0, 0])

        # Frontal key light, aligned with the tracking camera. Kept frontal so the
        # grayscale/base color codings keep the flat, head-on look of the real
        # Spotlight recordings, but brightened from the previous very dim level.
        self.mjcf_root.worldbody.add_light(
            name="thorax_light",
            pos=pos_offset,
            dir=view_direction,
            type=mj.mjtLightType.mjLIGHT_SPOT,
            diffuse=[0.5, 0.5, 0.5],
            # specular=[0.5, 0.5, 0.5],
            castshadow=True,
        )

        # Oblique fill light, tilted away from the view axis. A purely frontal
        # light gives almost uniform brightness on surfaces facing the camera, so
        # it carries little shape information; this oblique light makes the matte
        # shading pass vary with surface orientation (shape-from-shading), turning
        # the shading buffer into a usable geometry cue. Directional so the angle
        # is uniform across the fly; no shadow casting to keep the cue clean.
        oblique_angle_rad = np.deg2rad(45.0)
        oblique_direction = np.cos(oblique_angle_rad) * np.asarray(
            view_direction
        ) + np.sin(oblique_angle_rad) * np.asarray(camera_right)
        oblique_direction = oblique_direction / np.linalg.norm(oblique_direction)
        self.mjcf_root.worldbody.add_light(
            name="thorax_oblique_light",
            pos=pos_offset,
            dir=oblique_direction.tolist(),
            type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.6, 0.6, 0.6],
            castshadow=False,
        )
        return cam

class FlyBodyForRendering(FlyBody, FlyForRendering):
    def __init__(self, *args, **kwargs):
        self.all_material_names = []
        super().__init__(*args, **kwargs)
        self.sensorized_body_segments = []
    
    def add_joints(self, skeleton, neutral_pose):
        super().add_joints(skeleton, neutral_pose)
        
        # make the tarsus and claw segments a bit stiffer
        for jointdof, joint in self.jointdof_to_mjcfjoint.items():
            if "tarsus" in jointdof.name:
                joint.stiffness *= 5.0

        
class SingleFlySimulationForRendering(Simulation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.world.fly_lookup) == 1, "Expecting exactly one fly in the simulation."
        # Lookup key depends on whether use_flybody is set ("flybody" vs "nmf"),
        # so derive it from the lookup rather than hard-coding.
        self._fly_name = next(iter(self.world.fly_lookup))
        assert len(self.world.fly_lookup[self._fly_name].cameraname_to_mjcfcamera)
        self._map_internal_meshstatesensor_ids()
        self._map_internal_cardinalsensor_ids()
        self._map_internal_material_ids()

    
    def _map_internal_meshstatesensor_ids(self) -> None:
        internal_meshstatesensorids_by_fly = defaultdict(list)
        for fly_name, fly in self.world.fly_lookup.items():
            if len(fly.body_segment_sensor_lookup) == 0:
                continue
            for segment_name in fly.sensorized_body_segments:
                sensor = fly.body_segment_sensor_lookup["pos_atparent"][segment_name]
                internal_id = mj.mj_name2id(
                    self.mj_model, mj.mjtObj.mjOBJ_SENSOR, sensor.name
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
        Float[np.ndarray, "n_segments(69) 3"],
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
                    self.mj_model, mj.mjtObj.mjOBJ_SENSOR, sensor.name
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
    
    def _map_internal_material_ids(self) -> None:
        self._internal_matid_lookup = defaultdict(dict)
        for fly_name, fly in self.world.fly_lookup.items():
            for mat in fly.mjcf_root.materials:
                # After attaching the fly to the world, MjSpec has already
                # namespaced the fly's material names (e.g. "nmf/headthorax"),
                # so mat.name is the compiled name -- do not prefix it again.
                internal_id = mj.mj_name2id(
                    self.mj_model, mj.mjtObj.mjOBJ_MATERIAL, mat.name
                )
                assert internal_id != -1, f"Material {mat.name!r} not found in compiled model"
                self._internal_matid_lookup[fly_name][mat.name] = np.array(internal_id, dtype=np.int32)

    def colorize(self, fly_name:str, material_lookup:dict) -> None:
        if self._internal_matid_lookup is None:
            raise ValueError("Material lookup has not been set. Call _parse_visual_config() first to set the material lookup based on a visual config path.")
        for target, mat_name in material_lookup.items():
            # Standard Fly lookup uses body segments as keys.
            if hasattr(target, "name"):
                body_name = target.name
                self.mj_model.geom(f"{fly_name}/{body_name}").matid = self._internal_matid_lookup[fly_name][f"{fly_name}/{mat_name}"]
                continue

            # Flybody lookup uses geom names as keys.
            if isinstance(target, str):
                self.mj_model.geom(f"{fly_name}/{target}").matid = self._internal_matid_lookup[fly_name][f"{fly_name}/{mat_name}"]
                continue

            raise TypeError(f"Unsupported material lookup key type: {type(target)}")

def get_skeleton(use_flybody: bool):
    if use_flybody:
        axis_order = FlyBodyAxisOrder.YAW_ROLL_PITCH
        articulated_joints = FlyBodyJointPreset.LEGS_ONLY
        skeleton = FlyBodySkeleton(axis_order=axis_order, joint_preset=articulated_joints)
    else:
        axis_order = AxisOrder.YAW_PITCH_ROLL
        articulated_joints = JointPreset.LEGS_ONLY
        skeleton = Skeleton(axis_order=axis_order, joint_preset=articulated_joints)
    return skeleton

def set_up_simulation(render_window_size, render_play_speed, render_fps, sim_timestep, visual_paths, start_pose, use_flybody,
                      cam_pos_offset_z_mm: float = -67.0, cam_fovy_deg: float = 5.0):

    # This controlls how strongly the actuators try to track the target joint angles
    actuator_gain = 150.0  # in uN*mm/rad (torque applied per angular discrepancy)

    if use_flybody:
        skeleton = get_skeleton(use_flybody=True)
        fly = FlyBodyForRendering(mujoco_globals_path=nofuse_global)
        fly.add_joints(skeleton, neutral_pose=start_pose)
        actuated_dofs = FlyBodyActuatedDOFPreset.LEGS_ACTIVE_ONLY
        actuated_dofs_list = skeleton.get_actuated_dofs_from_preset(actuated_dofs)

        fly.add_actuators(
            actuated_dofs_list,
            actuator_type=actuator_type
            )
        
        spawn_pos = [0, 0, 2.0]  # higher release thorax origin is located in a different place

    else:
        skeleton = get_skeleton(use_flybody=False)
        fly = FlyForRendering(mujoco_globals_path=nofuse_global) # if bodies are fused, head disapears and we cannot add an xbody sensor to it 
        fly.add_joints(skeleton, neutral_pose=start_pose)
        actuated_dofs = ActuatedDOFPreset.LEGS_ACTIVE_ONLY
        actuated_dofs_list = skeleton.get_actuated_dofs_from_preset(actuated_dofs)
        fly.add_actuators(
            actuated_dofs_list,
            actuator_type=actuator_type,
            kp=actuator_gain,
            neutral_input=start_pose,
        )

        spawn_pos = [0, 0, 0.7]  # center of thorax is at 0.7 mm above the ground

    fly.add_mesh_state_sensors()
    fly.add_cardinal_sensors()
    
    # NOTE: the offscreen framebuffer size (offwidth/offheight) and the physics
    # timestep are global options resolved from the top-level (world) spec that
    # gets compiled, so they are set on the world below, after the fly is
    # attached, rather than on the fly spec here.

    expected_fly_name = "flybody" if use_flybody else "nmf"
    assert fly.name == expected_fly_name, (
        f"Expecting fly name to be '{expected_fly_name}' (use_flybody={use_flybody}) "
        f"for consistency with sensors and data recording; got '{fly.name}'."
    )

    materials_lookups = []
    for visual_path in visual_paths:
        _, lookup = fly._parse_visuals_config(visual_path)
        materials_lookups.append(lookup)
        fly.colorize(visual_path)

    # Track the whole-fly center of mass with mode="trackcom". This is a built-in
    # camera mode that adds no body to the model, so the fly's dynamics are identical
    # to the untracked baseline (a separate mount body cannot be made massless under
    # the compiler's boundmass floor, so it would perturb the replay).
    #
    # To center the field of view on that COM, the camera is given an in-plane offset
    # equal to the COM's position relative to the thorax origin (the body the camera is
    # attached to). That puts the camera directly over the COM, so the COM projects to
    # the image center instead of sitting off-center (the thorax origin and the COM
    # differ, which is what pushed the COM low before). Only the calibrated distance (z)
    # and fovy set the scale. Postprocessing then rotates and crops about the thorax
    # center of mass (thorax_com, saved per frame) to produce the thorax-centered framing;
    # the telephoto FOV makes a rotate-with-body ("fixed") camera too pitch-sensitive, so
    # uprighting is done in postprocessing.
    fly_only_model = fly.mjcf_root.compile()
    fly_only_data = mj.MjData(fly_only_model)
    mj.mj_forward(fly_only_model, fly_only_data)
    thorax_body_id = mj.mj_name2id(fly_only_model, mj.mjtObj.mjOBJ_BODY, "c_thorax")
    assert thorax_body_id != -1, "c_thorax body not found when centering the camera."
    thorax_xmat = fly_only_data.xmat[thorax_body_id].reshape(3, 3)
    com_offset_in_thorax = thorax_xmat.T @ (
        fly_only_data.subtree_com[thorax_body_id] - fly_only_data.xpos[thorax_body_id]
    )

    cam_offset = (
        float(com_offset_in_thorax[0]),
        float(com_offset_in_thorax[1]),
        float(cam_pos_offset_z_mm),
    )
    cam_rot = Rotation3D(format="euler", values=(0, np.pi, -np.pi / 2))
    tracking_cam = fly.add_tracking_camera(
        pos_offset=cam_offset, rotation=cam_rot, fovy=float(cam_fovy_deg), mode="trackcom",
    )
    fly.add_leg_adhesion()

    spawn_rot = Rotation3D(format="quat", values=[1, 0, 0, 0])  # no rotation

    world = SpotlightArena()
    world.add_fly(fly, spawn_pos, spawn_rot,
                  bodysegs_with_ground_contact = FlyBodyContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD if use_flybody else ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD)

    # Offscreen framebuffer size and physics timestep are global options taken
    # from the compiled (world) spec, so set them here after the fly is attached.
    world.mjcf_root.visual.global_.offwidth = render_window_size[0]
    world.mjcf_root.visual.global_.offheight = render_window_size[1]
    world.mjcf_root.option.timestep = sim_timestep

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

    return sim, renderers, tracking_cam, materials_lookups


def run_neuromechfly_simulation(
    trajectories_interp,
    render_window_size,
    render_play_speed,
    render_fps,
    sim_timestep,
    min_sim_duration_sec,
    visual_paths,
    max_sim_steps=None,
    render_depth=False,
    use_flybody=False,
    cam_pos_offset_z_mm: float = -67.0,
    cam_fovy_deg: float = 5.0,
):

    if use_flybody:
        axis_order = FlyBodyAxisOrder.YAW_ROLL_PITCH
    else:
        axis_order = AxisOrder.YAW_PITCH_ROLL        
    skeleton = get_skeleton(use_flybody=use_flybody)
    
    joint_angles_rad_dict = {}
    for joint, traj in zip(skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY), trajectories_interp[0]):
        joint_angles_rad_dict[joint.name] = traj

    start_pose = KinematicPose(
        joint_angles_rad_dict=joint_angles_rad_dict,
        axis_order=axis_order,
        mirror_left2right=False,
    )

    sim, renderers, cam, materials_lookups = set_up_simulation(
        render_window_size, render_play_speed, render_fps, sim_timestep, visual_paths, start_pose, use_flybody,
        cam_pos_offset_z_mm=cam_pos_offset_z_mm, cam_fovy_deg=cam_fovy_deg,
    )

    if render_depth:
        depth_renderer = Renderer(
            sim.mj_model,
            cam,
            camera_res=render_window_size,
            playback_speed=render_play_speed,
            output_fps=render_fps * render_play_speed,
            render_rgb=False,
            render_depth=True
        )
    # render the segment id as it will be used as the training set for segmentation task
    segmentid_renderer = Renderer(
        sim.mj_model,
        cam,
        camera_res=render_window_size,
        playback_speed=render_play_speed,
        output_fps=render_fps * render_play_speed,
        render_rgb=False,
        render_segmentation=True
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
    # list[ndarray of shape (3,)], thorax center of mass in world coordinates (the
    # thorax body's own inertial position). Postprocessing rotates and crops about its
    # projection so the processed fly is centered on the thorax, matching the real
    # Spotlight images' thorax-centered framing.
    thorax_com_hist = []

    fly_name = sim._fly_name
    sensorized_body_segments = sim.world.fly_lookup[fly_name].sensorized_body_segments
    base_pose_idx = sensorized_body_segments.index("c_thorax")  # used to track fly base position and flipping

    # Body the tracking camera follows (the thorax). Found by camera mode -- track(1)
    # or trackcom(2) -- so this does not depend on the camera's (possibly prefixed)
    # name. We save this body's center of mass (xipos, its inertial position) as the
    # point postprocessing centers on.
    _tracking_cam_ids = [
        c for c in range(sim.mj_model.ncam) if int(sim.mj_model.cam_mode[c]) in (1, 2)
    ]
    assert _tracking_cam_ids, "No tracking camera found; cannot record the tracked body COM."
    com_tracked_body_id = int(sim.mj_model.cam_bodyid[_tracking_cam_ids[0]])

    sim.warmup()
    # Simulation loop
    sim.set_leg_adhesion_states(fly_name, 0.5*np.ones(6, dtype=bool))
    for sim_frame_id in trange(trajectories_interp.shape[0], disable=None):
        # Stop if we have reached the desired number of simulation steps (for testing)
        if max_sim_steps is not None and sim_frame_id >= max_sim_steps:
            break

        # Step physics simulation
        sim.set_actuator_inputs(fly_name,
                                actuator_type,
                                trajectories_interp[sim_frame_id])
        sim.step()

        # Render all buffers on the frames flygym's renderer decides are due.
        # The first renderer's render_as_needed() return value is the single
        # timing gate; when it renders a frame, the other renderers (which share
        # the same output_fps and thus the same schedule) render on the same
        # frame and we record data. This avoids reaching into flygym's private
        # render-timing attributes, which the v2 Renderer no longer exposes.
        sim.colorize(fly_name, materials_lookups[0])  # appearance for renderer 0
        if not renderers[0].render_as_needed(sim.mj_data):
            continue  # not a render frame; skip rendering and data recording
        for i in range(1, len(renderers)):
            sim.colorize(fly_name, materials_lookups[i])  # match renderer i's color coding
            renderers[i].render_as_needed(sim.mj_data)
        if render_depth:
            depth_renderer.render_as_needed(sim.mj_data)
        segmentid_renderer.render_as_needed(sim.mj_data)
        

        # Store timestamps
        timestamps_hist.append(sim.time)

        # Store joint angles
        joints_angles_hist.append(sim.get_joint_angles(fly_name))

        # Store mesh states (from sensors added to each body segment)
        state_sensor_data = sim.get_mesh_state_info(fly_name)
        body_segment_state_hists["pos_atparent"].append(state_sensor_data)
        # for sensor_type, sensor_info in body_segment_sensor_lookup.items():
        #     num_sensors = len(sensor_info["segments_list"])
        #     bound_sensors = sensor_info["bound_sensors_list"]
        #     sensor_readings = bound_sensors.sensordata.copy()
        #     sensor_readings = sensor_readings.reshape((num_sensors, -1))
        #     body_segment_state_hists[sensor_type].append(sensor_readings)

        # Store global body segment states (from MjData)
        pos_global = sim.get_body_positions(fly_name)
        quat_global = sim.get_body_rotations(fly_name)
        body_segment_state_hists["pos_global"].append(pos_global)
        body_segment_state_hists["quat_global"].append(quat_global)

        # Store forward vector
        cardinal_vectors = sim.get_cardinal_vectors(fly_name)
        cardinal_vectors_hist.append(cardinal_vectors)  # forward/left/up vectors in world coordinates

        # Store camera matrix
        cam_xmat = renderers[0].get_camera_matrix(cam, sim.mj_data, sim.mj_model)  # assuming all renderers use the same camera
        camera_matrix_hist.append(cam_xmat)

        # Store fly base position
        fly_base_pos_hist.append(pos_global[base_pose_idx])

        # Store the thorax's own center of mass (the postprocessing pivot) in world
        # coordinates. xipos is updated by sim.step().
        thorax_com_hist.append(np.array(sim.mj_data.xipos[com_tracked_body_id], dtype=np.float32))

        # Check if the fly has flipped over
        up_vector_z_component = cardinal_vectors[2, 2]
        if up_vector_z_component < 0:
            print(f"Fly flipped over at frame {sim_frame_id}. Stopping simulation.")
            break

    final_simulated_time_sec = sim.time

    if render_depth:
        renderers.append(depth_renderer)
    renderers.append(segmentid_renderer)

    if final_simulated_time_sec < min_sim_duration_sec or sim_frame_id == 0:
        print(
            f"Simulation failed too early at {final_simulated_time_sec:.3f} sec. "
            f"Discarding results."
        )
        return renderers, None

    all_dofs = ["-".join([j.parent.name, j.child.name, j.axis.name.lower()])
                            for j in sim.world.fly_lookup[fly_name].get_actuated_jointdofs_order(actuator_type)]

    all_bodies = [b.name for b in sim.world.fly_lookup[fly_name].bodyseg_to_mjcfbody]
    all_world_geoms = [sim.mj_model.geom(i).name for i in range(sim.mj_model.ngeom)]

    import json
    
    # Process segmentation maps to explicitly set background to 0 and shift others by 1.
    # flygym-v2's Renderer returns segmentation in `segmentation_frames` as int32
    # object ids (channel 0 of MuJoCo's segmentation output) with background == -1.
    seg_maps_raw = np.array(
        segmentid_renderer.segmentation_frames[f"{fly_name}/trackcam"], dtype=np.int32
    )  # (n_frames, height, width), background == -1
    seg_maps = seg_maps_raw + 1  # background -> 0, geom id g -> g + 1
    assert seg_maps.max() < 256, (
        f"Segmentation ids exceed uint8 range (max {seg_maps.max()}); "
        "store segmentation_maps as uint16 instead."
    )
    seg_maps = seg_maps.astype(np.uint8)

    seg_mapping = {"Background": 0}
    for i, body in enumerate(all_world_geoms):
        seg_mapping[body] = i + 1
    
    segmentation_labels_json = json.dumps(seg_mapping)

    hist_dict = {
        "values": {
            "timestamp": timestamps_hist,
            "joint_angles": joints_angles_hist,
            "body_seg_states": body_segment_state_hists,
            "cardinal_vectors": cardinal_vectors_hist,
            "camera_matrix": camera_matrix_hist,
            "fly_base_pos": fly_base_pos_hist,
            "thorax_com": thorax_com_hist,
            "segmentation_maps": seg_maps, # store the segmentation maps as an array of shape (n_frames, height, width) with integer values corresponding to body ids in the simulation
        },
        "keys": {
            "joint_angles": all_dofs,
            "sensorized_body_segments": sensorized_body_segments,
            "all_bodies": all_bodies,
            "segmentation_labels": segmentation_labels_json, # Explicit JSON dictionary mapping string names to integer IDs
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
        for joint in hist_dict["keys"]["joint_angles"]:
            parent, child, axis = joint.split("-")
            kchain_name, link_name = parse_nmf_joint_seg(parent, child, axis)
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

        # Segmentation maps
        h5_file.create_dataset(
            "segmentation_maps",
            data=hist_dict["values"]["segmentation_maps"],
            dtype="uint8",
        )
        h5_file["segmentation_maps"].attrs["description"] = (
            "Segmentation maps rendered during the simulation. This dataset has shape "
            "(n_timesteps, height, width) and contains integer values corresponding to "
            "body ids in the simulation. The mapping from body ids to body names is given "
            "in the 'keys' attribute under 'segmentation_labels'."
        )
        h5_file["segmentation_maps"].attrs["keys"] = hist_dict["keys"]["segmentation_labels"]

        # Body segment states
        body_seg_group = h5_file.create_group("body_segment_states")
        segments_order = hist_dict["keys"]["sensorized_body_segments"]

        # Process each sensor type in the hist_dict
        for sensor_type, sensor_data_list in hist_dict["values"][
            "body_seg_states"
        ].items():
            if not sensor_data_list:  # Skip empty sensor data
                continue
            sensor_data_arr = np.array(sensor_data_list, dtype="float32")
            if sensor_data_arr.size == 0:
                continue
            assert sensor_data_arr.shape[0] == n_timesteps, (
                f"Expected {n_timesteps} timesteps for sensor type '{sensor_type}', "
                f"but got {sensor_data_arr.shape[0]}"
            )
            assert sensor_data_arr.shape[1] == len(segments_order), (
                f"Expected {len(segments_order)} segments for sensor type '{sensor_type}', "
                f"but got {sensor_data_arr.shape[1]}"
            )

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

        # Thorax center of mass (the postprocessing pivot, for thorax-centered framing).
        thorax_com_arr = np.array(hist_dict["values"]["thorax_com"], dtype="float32")
        thorax_com_ds = h5_file.create_dataset("thorax_com", data=thorax_com_arr, dtype="float32")
        thorax_com_ds.attrs["keys"] = ["x", "y", "z"]
        thorax_com_ds.attrs["units"] = "mm"
        thorax_com_ds.attrs["description"] = (
            "Thorax center of mass (the thorax body's inertial position) in global "
            "coordinates, per frame, shape (n_timesteps, 3). Postprocessing rotates and "
            "crops about its projection so the processed fly is centered on the thorax."
        )


def simulate_one_segment(
    kinematic_recording_segment: pd.DataFrame,
    output_dir: Path,
    input_timestep: float,
    sim_timestep: float,
    visual_paths: list[Path],
    output_data_freq: int = 300,
    render_play_speed: float = 0.1,
    render_window_size=(912, 912),
    min_sim_duration_sec: float = 0.2,
    max_sim_steps: int | None = None,
    render_depth: bool = False,
    use_flybody: bool = False,
    calibration_path: str | Path | None = None,
    source_format: str = "aymanns2022",
) -> tuple[bool, list[str]]:
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
            rendering the simulation. Both dimensions must be a multiple of 16
            to avoid FFMPEG automatically padding the output video, which causes
            shape mismatches with the unpadded segmentation maps.
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

    if render_window_size[0] % 16 != 0 or render_window_size[1] % 16 != 0:
        raise ValueError(
            f"render_window_size {render_window_size} must be a multiple of 16 in both "
            "dimensions to prevent FFMPEG from automatically padding the output video. "
            "Padding causes shape mismatches with the unpadded segmentation maps."
        )

    cam_pos_offset_z_mm = -67.0
    cam_fovy_deg = 5.0
    if calibration_path is not None:
        from poseforge.neuromechfly.camera_calibration import load_camera_config
        cfg = load_camera_config(calibration_path)
        cam_pos_offset_z_mm = cfg.pos_offset_z_mm
        cam_fovy_deg = cfg.fovy_deg
        print(
            f"[camera] using calibration {Path(calibration_path).name}: "
            f"fovy={cam_fovy_deg:.4f} deg, pos_offset_z={cam_pos_offset_z_mm:.3f} mm "
            f"(target {cfg.pixel_per_mm_at_sample:.2f} px/mm at sample, "
            f"render_size={cfg.render_size})"
        )
        if cfg.render_size != render_window_size[0] or cfg.render_size != render_window_size[1]:
            print(
                f"[camera] WARNING: calibration render_size={cfg.render_size} does not match "
                f"render_window_size={render_window_size}; the px/mm match assumes the renders "
                f"are at the same resolution as the postprocessed experimental crop."
            )

    if use_flybody:
        axis_order = FlyBodyAxisOrder.YAW_ROLL_PITCH
        articulated_joints = FlyBodyJointPreset.LEGS_ONLY
        actuated_dofs = FlyBodyActuatedDOFPreset.LEGS_ACTIVE_ONLY
        skeleton = FlyBodySkeleton(axis_order=axis_order, joint_preset=articulated_joints)
    else:
        axis_order = AxisOrder.YAW_PITCH_ROLL
        articulated_joints = JointPreset.LEGS_ONLY
        actuated_dofs = ActuatedDOFPreset.LEGS_ACTIVE_ONLY
        skeleton = Skeleton(axis_order=axis_order, joint_preset=articulated_joints)

    # Interpolate the trajectories to match the simulation timestep
    actuated_joints_list = skeleton.get_actuated_dofs_from_preset(actuated_dofs)
    trajectories_interp, interp_factor = interpolate_trajectories(
        kinematic_recording_segment, input_timestep, sim_timestep, actuated_joints_list, use_flybody,
        source_format=source_format,
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
        render_depth=render_depth,
        use_flybody=use_flybody,
        cam_pos_offset_z_mm=cam_pos_offset_z_mm,
        cam_fovy_deg=cam_fovy_deg,
    )

    # Do nothing if simulation failed before the minimum required duration is reached
    if hist_dict is None:
        # Save rendered frames as a video
        for i, renderer in enumerate(renderers):
            video_path = output_dir / f"nmf_sim_render_colorcode_{i}.mp4"
            renderer.save_video(video_path)
        return False, None

    # Save kinematic states as Pandas DataFrame
    h5_path = output_dir / "simulation_data.h5"
    make_simulation_data_h5(hist_dict, h5_path)

    from poseforge.neuromechfly import geometry_buffers

    # Save the RGB color-coding renders (one per visual path), including the
    # uniform-albedo "shading" set when it is among the visual paths.
    render_filenames = []
    for i, (renderer, visual_path) in enumerate(zip(renderers, visual_paths)):
        suffix = visual_path.stem if isinstance(visual_path, Path) else visual_path
        render_filename = f"nmf_sim_render_{suffix}.mp4"
        renderer.save_video(output_dir / render_filename)
        render_filenames.append(render_filename)

    # Save the derived geometry-prior buffers. The depth and segmentation
    # renderers are appended to `renderers` by run_neuromechfly_simulation
    # (segmentation last, depth just before it when render_depth is True); their
    # raw buffers live in `.depth_frames` / `.segmentation_frames`, not in video.
    fly_name = "flybody" if use_flybody else "nmf"
    cam_key = f"{fly_name}/trackcam"
    buffer_fps = renderers[0].output_fps

    segmentation_raw = np.array(
        renderers[-1].segmentation_frames[cam_key], dtype=np.int32
    )  # (n_frames, height, width), background == -1
    foreground_mask = segmentation_raw != -1  # (n_frames, height, width)

    segmentation_labels = segmentation_raw + 1  # background -> 0
    label_palette = geometry_buffers.build_label_palette(
        int(segmentation_labels.max()) + 1
    )
    segmentation_rgb = np.stack(
        [
            geometry_buffers.colorize_segmentation(frame, palette=label_palette)
            for frame in segmentation_labels
        ]
    )
    geometry_buffers.save_video(
        segmentation_rgb, output_dir / "nmf_sim_segment_id.mp4", fps=buffer_fps
    )
    render_filenames.append("nmf_sim_segment_id.mp4")

    if render_depth:
        depth_stack = np.array(
            renderers[-2].depth_frames[cam_key], dtype=np.float32
        )  # (n_frames, height, width), metric meters

        depth_rgb = np.stack(
            [
                geometry_buffers.colorize_depth(depth, mask)
                for depth, mask in zip(depth_stack, foreground_mask)
            ]
        )
        geometry_buffers.save_video(
            depth_rgb, output_dir / "nmf_sim_depth.mp4", fps=buffer_fps
        )
        render_filenames.append("nmf_sim_depth.mp4")

        normals_rgb = np.stack(
            [
                geometry_buffers.encode_normals_to_rgb(
                    geometry_buffers.depth_to_surface_normals(depth, cam_fovy_deg, mask),
                    mask,
                )
                for depth, mask in zip(depth_stack, foreground_mask)
            ]
        )
        geometry_buffers.save_video(
            normals_rgb, output_dir / "nmf_sim_normals.mp4", fps=buffer_fps
        )
        render_filenames.append("nmf_sim_normals.mp4")

    return True, render_filenames