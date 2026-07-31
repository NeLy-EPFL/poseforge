#!/usr/bin/env python
"""Replay retained IK periods in FlyGym and render the result.

For each selected period in an IK/FK-augmented periods `.h5` (see
`solve_ik.py`), feeds that period's `ik_dofangles_rad` (42 leg-only DOFs) as
position-actuator targets to a `NeuroMechFly` model and renders the
resulting simulation from two cameras: an oblique tracking camera (matching
`make_videos.py`'s synthetic 3D IK/FK panel's viewing angle) and a bottom-up
camera matching the real Spotlight rig, which films through the floor (only
its segmentation ("segid") view is saved; see below). The fly's overall
position/orientation is not prescribed: it emerges from the physics as the
leg actuators push against the ground, exactly as in
`flygym/tutorials/2_replaying_experimental_recordings.ipynb`.

As in that tutorial, the control frequency (how often actuator targets are
updated) is different from, and much lower than, the physics simulation
frequency: physics always steps at `PHYSICS_TIMESTEP` (1e-4 s, FlyGym's own
default), while actuator targets update once per recorded frame, at each
trial's own recording rate (`behavior_fps` in
`metadata/experiment_parameters.yaml`, 396 Hz for this dataset) held
constant (zero-order hold) across however many physics steps that spans.
Unlike the tutorial's demo data, no smoothing/upsampling of the target
angles themselves is needed here, since we're not changing their rate, only
holding each one across multiple physics steps.

Also saves a segmentation ("segid") video and an HDF5 file of simulation
data, in the same schema as `poseforge/neuromechfly/scripts/run_simulation.py`
(flygymv2 branch)'s `make_simulation_data_h5`, so downstream code built for
that format can load this data directly. One deviation: DOF names use
FlyGym's own `"{parent}-{child}-{axis}"` convention rather than that
script's Aymanns-et-al.-2022 relabeling (`parse_nmf_joint_seg`), since the
latter depends on kinematic-prior-pipeline constants not present here.

With `--combine-panels` (on by default), also builds a 4-panel video per
period: top-left/top-right reuse `make_videos.py`'s 2D pose and synthetic
3D IK/FK panels; bottom-left/bottom-right are this script's own segid and
tracking-camera renders.

Runs on the CPU (`flygym.Simulation`, not the GPU/warp backend).

Usage:
    python replay_in_flygym.py \
        bulk_data/prior-2dinvkin/sleap/lm_ported_score/periods_ikfk.h5 \
        bulk_data/prior-2dinvkin/sleap/lm_ported_score/flygym_replays \
        --periods-per-trial 1 --playback-speed 0.2
"""

import json
import shutil
import time
import zipfile
from colorsys import hsv_to_rgb
from pathlib import Path

import cv2
import h5py
import imageio.v3 as iio
import mujoco as mj
import numpy as np
import tyro
import yaml
from flygym import Renderer, Simulation
from flygym.anatomy import (
    ActuatedDOFPreset,
    AxisOrder,
    BodySegment,
    JointDOF,
    JointPreset,
    Skeleton,
)
from flygym.compose import (
    ActuatorType,
    FlatGroundWorld,
    KinematicPosePreset,
    NeuroMechFly,
)
from flygym.utils.math import Rotation3D
from joblib import Parallel, delayed
from loguru import logger

from poseforge.prior2d.scripts.make_videos import (
    ALIGNED_VIDEO_RELPATH,
    build_period_frames,
    build_work_item,
    compute_style_context,
    select_longest_periods,
)
from poseforge.prior2d.scripts.solve_ik import BODY_PLAN_PATH, load_body_plan

DATA_ROOT = Path("/mnt/upramdya_data/VAS/poseforge_paper_data")
METADATA_ZIP_RELPATH = Path("metadata.zip")
EXPERIMENT_PARAMETERS_YAML_NAME = "metadata/experiment_parameters.yaml"
NOFUSE_GLOBALS_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "mujoco_globals_nofuse.yaml"
)

# FlyGym's own default (see mujoco_globals.yaml); a much coarser physics step
# (e.g. matching the ~396 Hz control/recording rate directly) makes the
# contact-rich leg/ground dynamics numerically unstable.
PHYSICS_TIMESTEP = 1e-4  # seconds

SPAWN_POSITION = (0.0, 0.0, 0.7)  # thorax center, mm above the ground
SPAWN_ROTATION = Rotation3D(format="quat", values=(1.0, 0.0, 0.0, 0.0))
# The flygym tutorial's own default (150) is tuned for its own, comparatively
# clean SeqIKPy joint angles; our own IK's occasional per-frame noise makes
# such a stiff actuator overreact (a visibly "too rigid" whole-body jerk on
# noisy frames). flygym_demo.complex_terrain.common's own locomotion-tutorial
# default (45) is more compliant, but overcorrected the other way; values in
# between are being tried (150 -> 45 -> 100 -> 75 -> 50 -> 75 -> back to 50,
# settled) to find a good middle ground.
ACTUATOR_GAIN = 50.0  # uN*mm/rad
# The base API's own default is 1.0 (`fly.add_leg_adhesion()` with no
# argument); flygym_demo.complex_terrain.common.make_locomotion_fly's own
# locomotion-tutorial default (40, also the original NeuroMechFly paper's
# measured leg-adhesion force) is much stronger. Values in between are being
# tried (1.0 -> 0.7 -> 0.3 -> 0.0 -> back to 0.3, settled) to find a good
# middle ground.
ADHESION_GAIN = 0.3  # uN

# `add_tracking_camera`'s `mode="track"` (its own default) rigidly rotates
# the camera with the tracked body's full orientation (not just its
# position), verified empirically by comparing `mj_data.cam_xmat` at
# different spawn headings -- despite the method's own docstring describing
# "track" as position-only. A pos_offset/rotation defined once, relative to
# the thorax's own frame, therefore holds a constant *relative* viewing
# angle to the fly for the whole simulation, the same way `make_videos.py`'s
# synthetic 3D panel recomputes its camera every frame to track the fly's
# heading.
#
# Tracking camera: reuses `make_videos.py`'s own
# `compute_camera_orientation` (elevated, -30-deg pitch) so the two panels
# share a viewing angle, evaluated once for a synthetic fly facing local +X
# (this camera's pos_offset is in the thorax's own frame, where +X is
# "forward" at attachment time). `compute_camera_orientation`'s own azimuth
# is "45-deg right-front", but comparing this camera's render side by side
# with the IK/FK panel showed it looking from the fly's front-right while
# the panel looks from front-left -- so the azimuth is mirrored here
# (`forward - right_body` instead of `+`) to match:
#   fake_frame = [[1, 0.5, 0], [1, -0.5, 0], [-1, 0.5, 0], [-1, -0.5, 0]]
#   forward = ...  # see compute_camera_orientation; (0, 1, 2, 3) are lf/rf/lh/rh
#   right_body = [forward[1], -forward[0], 0]
#   azimuth_dir = normalize(forward - right_body)  # mirrored: left-front
#   pitch = radians(30); view_dir = normalize(cos(pitch) * azimuth_dir + sin(pitch) * z_hat)
#   right_axis = normalize(cross(view_dir, z_hat)); up_axis = cross(right_axis, view_dir)
#   pos_offset = 9.0 * view_dir  # mm; ~ the flygym tutorial default offset's own magnitude
#   xyaxes = (-right_axis, up_axis)  # camera's local Z (= cross(x, y)) must equal
#                                     # view_dir for it to look back toward the fly;
#                                     # cross(right_axis, up_axis) == -view_dir always
#                                     # (by construction), hence the negated right_axis.
TRACKING_CAM_POS_OFFSET_MM = (5.511351921262151, 5.511351921262151, 4.5)
TRACKING_CAM_ROTATION = Rotation3D(
    format="xyaxes",
    values=(
        -0.7071067811865476,
        0.7071067811865476,
        -0.0,
        -0.3535533905932738,
        -0.3535533905932738,
        0.8660254037844388,
    ),
)
TRACKING_CAM_FOVY_DEG = 30.0

# Bottom-up camera, looking through where the (otherwise opaque) floor would
# be toward the fly's underside (see `mark_ground_group`/
# `hide_ground_for_camera`), approximating the real Spotlight rig.
# `poseforge/neuromechfly/scripts/run_simulation.py` (flygymv2 branch) does
# the same with a camera 67 mm below the thorax and a 5-deg FOV, but those
# literal values don't transfer to this FlyGym version: its
# `add_tracking_camera` uses a different pos_offset/rotation convention (see
# that method's own docstring warning about the pre/post-MjSpec-migration
# offset frame change). These values are re-tuned from scratch for this
# version instead, verified by rendering test frames.
#
# This camera is a child of `bottom_cam_anchor` (see
# `add_bottom_camera_anchor`), a world-fixed-orientation mocap body, not of
# the fly itself -- so `xyaxes` is relative to the *world* frame, which
# coincides with the thorax's own frame only at spawn (yaw/pitch/roll all
# zero). `xyaxes=(0, 1, 0, 1, 0, 0)`: camera-local X = world +Y ("left" at
# spawn), camera-local Y (image "up") = world +X ("forward" at spawn), so
# the fly's head points toward the top of the frame at spawn -- matching
# the 2D pose panel's own "fly pointing up" framing (see
# `ZOOM_CROP_FRACTION`) -- and camera-local Z = cross(X, Y) = world -Z, so
# the camera looks along +Z (up through the floor) toward the fly, as
# intended. Because the anchor never rotates, this framing holds regardless
# of how much the fly yaws/pitches/rolls after spawn (unlike attaching
# directly to the fly, where any yaw would spin the image -- see
# `add_bottom_camera_anchor`).
BOTTOM_CAM_POS_OFFSET_MM = (0.0, 0.0, -20.0)
BOTTOM_CAM_ROTATION = Rotation3D(format="xyaxes", values=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0))
BOTTOM_CAM_FOVY_DEG = 30.0
# Crop the segid render to this fraction of its width/height (centered),
# then rescale back up, to roughly match the 2D pose panel's own apparent
# fly scale (see module docstring's `--combine-panels`).
ZOOM_CROP_FRACTION = 0.55
# Shift the crop window down by this fraction of the crop height, so the
# fly appears higher up in the final (rescaled) frame -- matching where it
# tends to sit in the 2D pose panel -- rather than dead-center.
ZOOM_CROP_Y_OFFSET_FRACTION = 0.12

THORAX_SEGMENT = BodySegment("c_thorax")
# Thorax-local x/y/z axes, in that order; see `add_cardinal_vector_sensors`.
CARDINAL_AXES = ("x", "y", "z")
CARDINAL_LABELS = ("forward", "left", "up")

# The tracking/bottom videos and the interoperable h5 data are all captured
# densely, once per control frame (see `replay_period`). The segid video is
# a convenience visualization, not interoperable data, so it's downsampled
# to a conventional video frame rate instead (see `select_sparse_indices`);
# the h5's own `segmentation_maps` stays dense.
SEGID_VIDEO_OUTPUT_FPS = 30.0


def load_control_freq_hz(trial_dir: Path) -> float:
    """Read a trial's recording frame rate, used as this replay's control freq.

    Args:
        trial_dir: Trial directory containing `metadata.zip`.

    Returns:
        `behavior_fps` from `metadata/experiment_parameters.yaml`, read
        directly out of `metadata.zip` (never extracted to disk, since
        `trial_dir` typically lives on a read-only mount).
    """
    with zipfile.ZipFile(trial_dir / METADATA_ZIP_RELPATH) as zf:
        raw_yaml = zf.read(EXPERIMENT_PARAMETERS_YAML_NAME)
    return float(yaml.safe_load(raw_yaml)["behavior_fps"])


def build_dof_reorder_index(
    dof_names: list[str], actuated_dof_order: list[JointDOF]
) -> np.ndarray:
    """Map body-plan DOF order to the order FlyGym's actuators expect.

    Both use the same `"{parent}-{child}-{axis}"` naming (verified to align
    exactly, including sign convention, since the body plan was exported
    from this same FlyGym model; see `flygym/scripts/export_model_for_quickik.py`).

    Args:
        dof_names: DOF names in `ik_dofangles_rad`'s order (see `solve_ik.py`).
        actuated_dof_order: `fly.get_actuated_jointdofs_order(actuator_type)`.

    Returns:
        Index array such that `ik_dofangles_rad[..., index]` is reordered to
        `actuated_dof_order`.
    """
    name_to_idx = {name: i for i, name in enumerate(dof_names)}
    index = [
        name_to_idx[f"{dof.parent.name}-{dof.child.name}-{dof.axis.value}"]
        for dof in actuated_dof_order
    ]
    return np.array(index, dtype=np.int64)


def add_body_position_sensors(
    world: FlatGroundWorld, fly: NeuroMechFly
) -> list[BodySegment]:
    """Add a global-frame position sensor to every body segment.

    Must be called after `world.add_fly`: sensors are added to the world's
    (already fly-attached) spec, referencing each segment's current,
    possibly-namespaced body name -- the same pattern FlyGym itself uses for
    ground-contact sensors (see `_add_ground_contact_sensors` in
    `flygym/compose/world/base_world.py`).

    Args:
        world: World with `fly` already spawned on it.
        fly: The fly to sensorize.

    Returns:
        Body segments in the order their sensors were added; also the order
        to read the sensors back in, and the "keys" order for the
        interoperable `body_segment_states` group.
    """
    segments = fly.get_bodysegs_order()
    for seg in segments:
        body = fly.bodyseg_to_mjcfbody[seg]
        world.mjcf_root.add_sensor(
            name=f"{seg.name}_pos_atparent",
            type=mj.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mj.mjtObj.mjOBJ_XBODY,
            objname=body.name,
        )
    return segments


def add_cardinal_vector_sensors(world: FlatGroundWorld, fly: NeuroMechFly) -> None:
    """Add thorax-local x/y/z axis sensors (-> forward/left/up in world frame).

    Must be called after `world.add_fly`; see `add_body_position_sensors`.
    """
    body = fly.bodyseg_to_mjcfbody[THORAX_SEGMENT]
    for axis in CARDINAL_AXES:
        world.mjcf_root.add_sensor(
            name=f"thorax_{axis}axis",
            type=getattr(mj.mjtSensor, f"mjSENS_FRAME{axis.upper()}AXIS"),
            objtype=mj.mjtObj.mjOBJ_XBODY,
            objname=body.name,
        )


# Fly geoms use groups 0 (normal) and 2 (hidden from the eye camera); marker
# geoms (unused here) use 1 or 4. MuJoCo's default scene option only shows
# groups 0-2 (`mjv_defaultOption`), so the ground needs one of those to stay
# visible by default for the tracking camera; group 1 is otherwise unused in
# this script's setup, and is hidden per-camera for the bottom camera (see
# `hide_ground_for_camera`).
GROUND_GEOM_GROUP = 1


def mark_ground_group(world: FlatGroundWorld) -> None:
    """Assign the ground plane to `GROUND_GEOM_GROUP`.

    The tracking camera renders every group (its default scene option), so
    the ground stays visible there, exactly as in the plain `FlatGroundWorld`.
    The bottom camera hides this one group (see `hide_ground_for_camera`) so
    it can see the fly from underneath, through where the (otherwise opaque)
    floor would be.
    """
    world.ground_geom.group = GROUND_GEOM_GROUP


def hide_ground_for_camera(renderer: Renderer) -> None:
    """Hide `GROUND_GEOM_GROUP` for one renderer's own scene option.

    Must be called after constructing `renderer`: its `__init__` always
    resets `scene_option` to MuJoCo's defaults (all groups visible) via
    `mj.mjv_defaultOption`, so any group visibility change has to happen
    after that, not by pre-configuring an option passed into the
    constructor.
    """
    renderer.scene_option.geomgroup[GROUND_GEOM_GROUP] = 0


def add_bottom_camera_anchor(world: FlatGroundWorld) -> mj.MjsBody:
    """Add a world-fixed-orientation mocap body to anchor the bottom camera.

    `add_tracking_camera`'s `mode="track"` rigidly rotates the camera with
    the tracked body's *full* orientation, not just its position (see
    `TRACKING_CAM_ROTATION`'s comment) -- fine for the oblique tracking
    camera, but wrong for the bottom camera: since it looks straight up
    along the fly's own vertical axis, any yaw the fly does (entirely
    normal while walking) spins the camera around its own view axis,
    rotating the whole image in-frame as the fly turns, rather than keeping
    a fixed "up" the way the real Spotlight camera (and the 2D pose panel
    it feeds) does.

    A mocap body's pose is driven directly by `mj_data.mocap_pos`/
    `mocap_quat`, with no physics of its own; updating only its position
    every physics step (see `replay_period`) and never its orientation
    keeps a camera rigidly attached to it translating with the fly while
    staying fixed in world orientation. Must be a direct child of the
    worldbody (a MuJoCo requirement for mocap bodies), so this must be
    called after `world.add_fly` (not before, unlike the fly's own
    cameras): attaching it beforehand would put it inside the fly's own
    (about-to-be-namespaced) subtree instead.
    """
    anchor = world.mjcf_root.worldbody.add_body(
        name="bottom_cam_anchor", pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)
    )
    anchor.mocap = True
    return anchor


def build_fly_and_world() -> tuple[
    NeuroMechFly,
    FlatGroundWorld,
    mj.MjsCamera,
    mj.MjsCamera,
    mj.MjsBody,
    list[BodySegment],
]:
    """Build the leg-actuated NeuroMechFly model and a flat-ground world.

    Mirrors `flygym/tutorials/2_replaying_experimental_recordings.ipynb`:
    legs-only skeleton, position actuators on the active leg DOFs, leg
    adhesion. Static-body fusion is disabled (`NOFUSE_GLOBALS_PATH`) so
    every body segment stays addressable by the position sensors added here
    (otherwise unarticulated segments like the head or wings disappear as
    distinct bodies).

    Returns:
        fly: The configured fly.
        world: A `FlatGroundWorld` with `fly` spawned on it.
        tracking_cam: Above/behind camera element, for `flygym.Renderer`.
        bottom_cam: Bottom-up camera element, for `flygym.Renderer`; a child
            of `bottom_cam_anchor`, not of the fly.
        bottom_cam_anchor: See `add_bottom_camera_anchor`.
        body_segments: See `add_body_position_sensors`.
    """
    fly = NeuroMechFly(mujoco_globals_path=NOFUSE_GLOBALS_PATH)
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY
    )
    fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)

    actuated_dofs = fly.skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    fly.add_actuators(
        actuated_dofs,
        actuator_type=ActuatorType.POSITION,
        kp=ACTUATOR_GAIN,
        neutral_input=KinematicPosePreset.NEUTRAL,
    )
    fly.add_leg_adhesion(gain=ADHESION_GAIN)
    fly.colorize()
    tracking_cam = fly.add_tracking_camera(
        pos_offset=TRACKING_CAM_POS_OFFSET_MM,
        rotation=TRACKING_CAM_ROTATION,
        fovy=TRACKING_CAM_FOVY_DEG,
    )

    world = FlatGroundWorld()
    mark_ground_group(world)
    world.add_fly(fly, SPAWN_POSITION, SPAWN_ROTATION)

    # Anchored on a mocap body (see `add_bottom_camera_anchor`), not on the
    # fly itself, so it translates with the fly without rotating with it.
    bottom_cam_anchor = add_bottom_camera_anchor(world)
    bottom_cam = bottom_cam_anchor.add_camera(
        name="bottomcam",
        pos=BOTTOM_CAM_POS_OFFSET_MM,
        fovy=BOTTOM_CAM_FOVY_DEG,
        **BOTTOM_CAM_ROTATION.as_kwargs(),
    )

    body_segments = add_body_position_sensors(world, fly)
    add_cardinal_vector_sensors(world, fly)

    return fly, world, tracking_cam, bottom_cam, bottom_cam_anchor, body_segments


def resolve_sensor_slice(mj_model: mj.MjModel, sensor_name: str) -> slice:
    """Resolve a named sensor's `mj_data.sensordata` slice in a compiled model."""
    sensor_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_SENSOR, sensor_name)
    start = mj_model.sensor_adr[sensor_id]
    dim = mj_model.sensor_dim[sensor_id]
    return slice(start, start + dim)


def id_to_rgb(segment_id: int) -> tuple[int, int, int]:
    """Deterministic false color for a segmentation id (0 is background/black).

    Hues are spaced by the golden-ratio conjugate so adjacent ids look
    visually distinct; used only for the human-viewable segmentation video,
    not the underlying id data.
    """
    if segment_id == 0:
        return (0, 0, 0)
    hue = (segment_id * 0.6180339887498949) % 1.0
    r, g, b = hsv_to_rgb(hue, 0.65, 0.95)
    return (round(r * 255), round(g * 255), round(b * 255))


def build_segmentation_label_mapping(mj_model: mj.MjModel) -> dict[str, int]:
    """Map every geom name to its segmentation id; `"Background"` is 0.

    Raw ids from `Renderer.segmentation_frames` are geom indices, with -1
    for background (see `flygym/rendering.py`); shifted by +1 here so 0 is
    reserved for background (see `shift_segmentation_maps`).
    """
    mapping = {"Background": 0}
    for i in range(mj_model.ngeom):
        mapping[mj_model.geom(i).name] = i + 1
    return mapping


def shift_segmentation_maps(raw_frames: list[np.ndarray]) -> np.ndarray:
    """Stack raw per-frame segmentation ids, shifting background from -1 to 0."""
    stacked = np.stack(raw_frames)
    return np.where(stacked != -1, stacked + 1, 0).astype(np.uint8)


def select_sparse_indices(
    sim_time: np.ndarray, output_fps: float, playback_speed: float
) -> np.ndarray:
    """Pick a sparser subset of a densely-sampled time series for video output.

    Greedily keeps a sample whenever at least `1 / (output_fps /
    playback_speed)` simulated seconds have elapsed since the last kept
    sample -- the same pacing `Renderer.render_as_needed` uses internally,
    just applied after the fact to data that was already captured at full
    density (see `SEGID_VIDEO_OUTPUT_FPS`).

    Args:
        sim_time: `(n_frames,)` simulation time, seconds, densely sampled
            (one entry per control frame).
        output_fps: Desired output video frame rate.
        playback_speed: Desired video playback speed relative to real time.

    Returns:
        Indices into `sim_time` to keep, so that encoding
        `data[indices]` at `fps=output_fps` plays back at `playback_speed`.
    """
    secs_between = 1.0 / (output_fps / playback_speed)
    selected = []
    next_time = -np.inf
    for i, t in enumerate(sim_time):
        if t >= next_time:
            selected.append(i)
            next_time = t + secs_between
    return np.array(selected, dtype=np.int64)


def colorize_segmentation_maps(segmentation_maps: np.ndarray) -> np.ndarray:
    """Palette-map segmentation ids (background 0) to RGB, for video encoding."""
    palette = np.array(
        [id_to_rgb(i) for i in range(int(segmentation_maps.max()) + 1)], dtype=np.uint8
    )
    return palette[segmentation_maps]


def zoom_crop_frames(
    frames: np.ndarray, crop_fraction: float, y_offset_fraction: float = 0.0
) -> np.ndarray:
    """Crop each frame to a `crop_fraction` window, then rescale back up.

    Args:
        frames: `(n_frames, height, width, 3)` uint8.
        crop_fraction: Fraction of width/height to keep.
        y_offset_fraction: Shifts the crop window down by this fraction of
            the crop height, moving the subject up in the rescaled output
            (e.g. `0.12` if the subject sits dead-center in `frames` but
            should appear a bit higher in the output); `0.0` keeps the crop
            vertically centered. Clamped so the crop window stays in bounds.

    Returns:
        Array of the same shape as `frames`.
    """
    n, height, width = frames.shape[:3]
    crop_h, crop_w = round(height * crop_fraction), round(width * crop_fraction)
    y0 = (height - crop_h) // 2 + round(y_offset_fraction * crop_h)
    y0 = max(0, min(y0, height - crop_h))
    x0 = (width - crop_w) // 2
    cropped = frames[:, y0 : y0 + crop_h, x0 : x0 + crop_w]
    return np.stack(
        [
            cv2.resize(cropped[i], (width, height), interpolation=cv2.INTER_LINEAR)
            for i in range(n)
        ]
    )


def rotate_frames_to_heading(frames: np.ndarray, yaw_deg: np.ndarray) -> np.ndarray:
    """Rotate each frame to counteract the fly's current yaw.

    `BOTTOM_CAM_ROTATION` is calibrated so the fly points "up" at yaw 0; as
    the fly turns (see the drift discussed for `replay_period`), it no
    longer does. This is a video-presentation-only fix (does not touch the
    h5's own, unrotated `segmentation_maps`): rotating each frame by its own
    `yaw_deg` around the image center keeps the fly pointing "up"
    throughout, since the camera is centered on the fly (see
    `add_bottom_camera_anchor`).

    Args:
        frames: `(n_frames, height, width, 3)` uint8.
        yaw_deg: `(n_frames,)` degrees; see module docstring's cardinal
            vectors for how this is derived from `forward`.

    Returns:
        Array of the same shape as `frames`.
    """
    n, height, width = frames.shape[:3]
    center = (width / 2, height / 2)
    return np.stack(
        [
            cv2.warpAffine(
                frames[i],
                cv2.getRotationMatrix2D(center, float(yaw_deg[i]), 1.0),
                (width, height),
            )
            for i in range(n)
        ]
    )


# Re-encoding at a sensible CRF (imageio's own `quality=8` default maps to
# CRF 10, a near-lossless setting that produces unnecessarily large files)
# plus `+faststart` (moves the moov atom to the front of the file) fixes
# both file size and playback in players that expect a streamable mp4
# (e.g. Slack's inline preview) but otherwise reject one with an
# end-of-file index.
VIDEO_OUTPUT_PARAMS = ["-crf", "23", "-movflags", "+faststart"]


def write_video(
    frames: np.ndarray | list[np.ndarray], output_path: Path, fps: float
) -> None:
    """Write `frames` (RGB, uint8) to `output_path` as an mp4."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        output_path,
        list(frames),
        fps=fps,
        codec="libx264",
        quality=None,
        output_params=VIDEO_OUTPUT_PARAMS,
    )


def save_simulation_data_h5(
    h5_path: Path,
    sim_time: np.ndarray,
    dof_angles: np.ndarray,
    dof_names: list[str],
    segmentation_maps: np.ndarray,
    segmentation_labels: dict[str, int],
    pos_atparent: np.ndarray,
    pos_global: np.ndarray,
    quat_global: np.ndarray,
    body_segment_names: list[str],
    cardinal_vectors: np.ndarray,
    camera_matrix: np.ndarray,
    fly_base_pos: np.ndarray,
) -> None:
    """Save one period's simulation data.

    Schema matches `poseforge/neuromechfly/scripts/run_simulation.py`
    (flygymv2 branch)'s `make_simulation_data_h5`, so downstream code built
    for that format can load this data directly (see module docstring for
    the one deviation: DOF naming).

    Args:
        h5_path: Where to save the `.h5` file.
        sim_time: `(n_frames,)` simulation time, seconds.
        dof_angles: `(n_frames, n_dofs)`, `fly.get_jointdofs_order()` order.
        dof_names: DOF names in `dof_angles`'s order.
        segmentation_maps: `(n_frames, height, width)` uint8 ids (bottom cam).
        segmentation_labels: Geom name -> id (see `build_segmentation_label_mapping`).
        pos_atparent: `(n_frames, n_segments, 3)`, mm.
        pos_global: `(n_frames, n_segments, 3)`, mm.
        quat_global: `(n_frames, n_segments, 4)`, (w, x, y, z).
        body_segment_names: Segment names, matching `pos_atparent`/`pos_global`/
            `quat_global`'s segment axis.
        cardinal_vectors: `(n_frames, 3, 3)`, axis 1 is forward/left/up (see
            `CARDINAL_LABELS`), axis 2 is x/y/z in world coordinates.
        camera_matrix: `(n_frames, 3, 4)`, bottom camera.
        fly_base_pos: `(n_frames, 3)`, thorax position in world coordinates, mm.
    """
    with h5py.File(h5_path, "w") as f:
        f.attrs["n_timesteps"] = len(sim_time)

        time_ds = f.create_dataset("sim_time", data=sim_time.astype("float32"))
        time_ds.attrs["units"] = "s"
        time_ds.attrs["description"] = "Time in the NeuroMechFly simulation"

        dof_ds = f.create_dataset("dof_angles", data=dof_angles.astype("float32"))
        dof_ds.attrs["keys"] = dof_names
        dof_ds.attrs["units"] = "radians"
        dof_ds.attrs["description"] = (
            "Angles of DoFs tracked in the simulation. Shape (n_timesteps, "
            "n_dofs); DoF order given by the 'keys' attribute."
        )

        seg_ds = f.create_dataset(
            "segmentation_maps", data=segmentation_maps, dtype="uint8"
        )
        seg_ds.attrs["keys"] = json.dumps(segmentation_labels)
        seg_ds.attrs["description"] = (
            "Segmentation maps rendered from the bottom camera. Shape "
            "(n_timesteps, height, width); integer values map to body/geom "
            "ids via the JSON dict in the 'keys' attribute."
        )

        body_group = f.create_group("body_segment_states")
        body_group.attrs["keys"] = body_segment_names
        body_group.attrs["description"] = (
            "Position (mm) and orientation (quaternion) of each body segment. "
            "'atparent' is MuJoCo's xbody frame (global position of the "
            "body's own joint-with-parent frame); 'global' is the body's own "
            "global frame (MuJoCo's xpos/xquat)."
        )
        pos_atparent_ds = body_group.create_dataset(
            "pos_atparent", data=pos_atparent.astype("float32")
        )
        pos_atparent_ds.attrs["keys"] = ["x", "y", "z"]
        pos_atparent_ds.attrs["units"] = "mm"

        pos_global_ds = body_group.create_dataset(
            "pos_global", data=pos_global.astype("float32")
        )
        pos_global_ds.attrs["keys"] = ["x", "y", "z"]
        pos_global_ds.attrs["units"] = "mm"

        quat_global_ds = body_group.create_dataset(
            "quat_global", data=quat_global.astype("float32")
        )
        quat_global_ds.attrs["keys"] = ["w", "x", "y", "z"]
        quat_global_ds.attrs["units"] = "quaternion"

        cardinal_group = f.create_group("cardinal_vectors")
        cardinal_group.attrs["keys"] = list(CARDINAL_LABELS)
        cardinal_group.attrs["description"] = (
            "Unit vectors pointing forward/left/up from the fly's thorax, in "
            "global coordinates. Shape (n_timesteps, 3) each."
        )
        for i, label in enumerate(CARDINAL_LABELS):
            ds = cardinal_group.create_dataset(
                label, data=cardinal_vectors[:, i, :].astype("float32")
            )
            ds.attrs["keys"] = ["x", "y", "z"]

        cam_ds = f.create_dataset("camera_matrix", data=camera_matrix.astype("float32"))
        cam_ds.attrs["description"] = (
            "3x4 camera matrix (bottom camera) for each frame "
            "(see https://en.wikipedia.org/wiki/Camera_matrix)."
        )

        base_pos_ds = f.create_dataset(
            "fly_base_pos", data=fly_base_pos.astype("float32")
        )
        base_pos_ds.attrs["keys"] = ["x", "y", "z"]
        base_pos_ds.attrs["units"] = "mm"
        base_pos_ds.attrs["description"] = "Thorax position in global coordinates."


def replay_period(
    fly: NeuroMechFly,
    world: FlatGroundWorld,
    tracking_cam: mj.MjsCamera,
    bottom_cam: mj.MjsCamera,
    bottom_cam_anchor: mj.MjsBody,
    body_segments: list[BodySegment],
    target_angles: np.ndarray,
    control_freq_hz: float,
    output_dir: Path,
    output_stem: str,
    video_height: int,
    playback_speed: float,
    video_item: dict | None,
) -> None:
    """Replay one period, rendering both cameras and saving interoperable data.

    Saves, under `output_dir`:

    - `{output_stem}_tracking.mp4`: normal color video, tracking camera,
      dense (one frame per control update, at `playback_speed`).
    - `{output_stem}_segid.mp4`: false-colored segmentation video, bottom
      camera, cropped/zoomed (see `ZOOM_CROP_FRACTION`) and downsampled to
      `SEGID_VIDEO_OUTPUT_FPS` (a convenience visualization, not
      interoperable data). This is the raw render, NOT rotated to counteract
      simulated heading drift (see `rotate_frames_to_heading`); only the
      combined video's bottom-left panel gets that rotation, since it's
      meant to visually track the top panels' fixed framing.
    - `{output_stem}_simulation_data.h5`: interoperable simulation data (see
      `save_simulation_data_h5`), dense, including every control frame's
      (uncropped) segmentation map (not just the ones kept in `_segid.mp4`).
    - `{output_stem}_combined.mp4`, if `video_item` is given: a 4-panel
      video (top-left/top-right: `make_videos.py`'s 2D pose and synthetic 3D
      IK/FK panels; bottom-left/bottom-right: this script's own segid and
      tracking-camera renders), dense, at `2 * video_height` per side. The
      bottom-left panel is rotated frame-by-frame to counteract simulated
      heading drift (see `rotate_frames_to_heading`), unlike the standalone
      `_segid.mp4` above.

    Args:
        fly, world: See `build_fly_and_world`.
        tracking_cam, bottom_cam, bottom_cam_anchor: See `build_fly_and_world`.
        body_segments: See `add_body_position_sensors`.
        target_angles: `(n_steps, n_actuated_dofs)` position-actuator targets,
            in `fly.get_actuated_jointdofs_order(ActuatorType.POSITION)` order,
            one recorded frame per control update.
        control_freq_hz: Actuator targets update at this rate (zero-order
            hold), independent of `PHYSICS_TIMESTEP`; see module docstring.
        output_dir: Directory to save this period's outputs to.
        output_stem: Filename stem for this period's outputs.
        video_height: Output height (and width; both cameras render square)
            in pixels, for every panel.
        playback_speed: Video playback speed relative to real time.
        video_item: `None` to skip the combined video, or a dict as returned
            by `make_videos.build_work_item` for this same period (its own
            `video_path` must exist).
    """
    camera_res = (video_height, video_height)
    sim = Simulation(world, timestep=PHYSICS_TIMESTEP)
    # `output_fps / playback_speed` must equal `control_freq_hz` so exactly one
    # frame is captured per control update regardless of playback_speed (see
    # `Renderer._secs_between_renders`); only the video's own metadata FPS
    # (and so its playback speed) changes.
    output_fps = control_freq_hz * playback_speed

    tracking_renderer = Renderer(
        sim.mj_model,
        tracking_cam,
        camera_res=camera_res,
        playback_speed=playback_speed,
        output_fps=output_fps,
    )
    bottom_renderer = Renderer(
        sim.mj_model,
        bottom_cam,
        camera_res=camera_res,
        playback_speed=playback_speed,
        output_fps=output_fps,
        render_rgb=False,
        render_segmentation=True,
    )
    hide_ground_for_camera(bottom_renderer)

    pos_atparent_slices = [
        resolve_sensor_slice(sim.mj_model, f"{seg.name}_pos_atparent")
        for seg in body_segments
    ]
    cardinal_slices = {
        axis: resolve_sensor_slice(sim.mj_model, f"thorax_{axis}axis")
        for axis in CARDINAL_AXES
    }
    body_segment_names = [seg.name for seg in body_segments]
    thorax_idx = body_segment_names.index(THORAX_SEGMENT.name)

    thorax_body_id = mj.mj_name2id(
        sim.mj_model, mj.mjtObj.mjOBJ_BODY, fly.bodyseg_to_mjcfbody[THORAX_SEGMENT].name
    )
    anchor_body_id = mj.mj_name2id(
        sim.mj_model, mj.mjtObj.mjOBJ_BODY, bottom_cam_anchor.name
    )
    anchor_mocap_id = sim.mj_model.body_mocapid[anchor_body_id]

    fly_name = fly.name
    sim.reset()
    sim.set_leg_adhesion_states(fly_name, np.ones(6, dtype=bool))
    sim.warmup()

    control_dt = 1.0 / control_freq_hz
    start_time = sim.time
    # Half a physics step of slack absorbs floating-point rounding in the
    # time comparison, without letting a whole extra physics step slip in.
    half_physics_step = 0.5 * PHYSICS_TIMESTEP

    sim_time_hist = []
    dof_angles_hist = []
    pos_atparent_hist = []
    pos_global_hist = []
    quat_global_hist = []
    cardinal_vectors_hist = []
    camera_matrix_hist = []
    fly_base_pos_hist = []

    wall_start = time.perf_counter()
    for step_idx in range(target_angles.shape[0]):
        sim.set_actuator_inputs(
            fly_name, ActuatorType.POSITION, target_angles[step_idx]
        )
        # Zero-order hold: step physics at its own fixed rate until this
        # frame's control interval has elapsed, then move to the next
        # target. Comparing against absolute elapsed time (rather than
        # counting a fixed number of substeps per frame) means rounding
        # doesn't accumulate, even though control_dt / PHYSICS_TIMESTEP
        # isn't an integer (e.g. ~25.25 at 396 Hz).
        target_time = start_time + (step_idx + 1) * control_dt
        while sim.time < target_time - half_physics_step:
            sim.step()
            # Keep the bottom camera's mocap anchor translating with the
            # thorax every physics step (never rotating it -- see
            # `add_bottom_camera_anchor`). `mj_step` doesn't itself
            # propagate a mocap body's new position into its (camera) child
            # frames; `mj_kinematics` is the cheap way to do that before
            # rendering, short of a second full step.
            sim.mj_data.mocap_pos[anchor_mocap_id] = sim.mj_data.xpos[thorax_body_id]
            mj.mj_kinematics(sim.mj_model, sim.mj_data)
            # Both renderers share output_fps/playback_speed, so they decide
            # to capture a frame at the same instants; bottom_renderer's
            # return value gates whether we also record non-video data.
            rendered = bottom_renderer.render_as_needed(sim.mj_data)
            tracking_renderer.render_as_needed(sim.mj_data)
            if not rendered:
                continue

            pos_global = sim.get_body_positions(fly_name)
            sim_time_hist.append(sim.time)
            dof_angles_hist.append(sim.get_joint_angles(fly_name))
            pos_atparent_hist.append(
                np.stack(
                    [sim.mj_data.sensordata[s].copy() for s in pos_atparent_slices]
                )
            )
            pos_global_hist.append(pos_global)
            quat_global_hist.append(sim.get_body_rotations(fly_name))
            cardinal_vectors_hist.append(
                np.stack(
                    [
                        sim.mj_data.sensordata[cardinal_slices[a]].copy()
                        for a in CARDINAL_AXES
                    ]
                )
            )
            camera_matrix_hist.append(
                bottom_renderer.get_camera_matrix(bottom_cam, sim.mj_data, sim.mj_model)
            )
            fly_base_pos_hist.append(pos_global[thorax_idx])

    wall_elapsed = time.perf_counter() - wall_start
    sim_elapsed = sim.time - start_time
    realtime_factor = sim_elapsed / wall_elapsed if wall_elapsed > 0 else float("nan")

    output_dir.mkdir(parents=True, exist_ok=True)
    # `Renderer.save_video` hardcodes its own `quality=8` (see
    # `VIDEO_OUTPUT_PARAMS`'s comment); passing `-crf` again here via
    # `output_params` overrides it, since ffmpeg takes the last value for a
    # repeated libx264 option.
    tracking_renderer.save_video(
        output_dir / f"{output_stem}_tracking.mp4", output_params=VIDEO_OUTPUT_PARAMS
    )

    # Dense (every control frame): the uncropped array goes into the
    # interoperable h5 below. The cropped/zoomed, colorized version is used
    # as-is (no rotation) for the standalone segid video (further subsampled
    # to SEGID_VIDEO_OUTPUT_FPS); a separately rotated-to-heading version is
    # used only for the combined video's bottom-left panel (see
    # `rotate_frames_to_heading`'s docstring for why).
    segmentation_maps = shift_segmentation_maps(
        bottom_renderer.segmentation_frames[bottom_cam.name]
    )
    segid_colorized = colorize_segmentation_maps(segmentation_maps)
    segid_rgb_dense = zoom_crop_frames(
        segid_colorized, ZOOM_CROP_FRACTION, ZOOM_CROP_Y_OFFSET_FRACTION
    )

    sim_time_arr = np.array(sim_time_hist)
    segid_indices = select_sparse_indices(
        sim_time_arr, SEGID_VIDEO_OUTPUT_FPS, playback_speed
    )
    write_video(
        segid_rgb_dense[segid_indices],
        output_dir / f"{output_stem}_segid.mp4",
        SEGID_VIDEO_OUTPUT_FPS,
    )

    dof_names = [
        f"{d.parent.name}-{d.child.name}-{d.axis.value}"
        for d in fly.get_jointdofs_order()
    ]
    save_simulation_data_h5(
        output_dir / f"{output_stem}_simulation_data.h5",
        sim_time=sim_time_arr,
        dof_angles=np.array(dof_angles_hist),
        dof_names=dof_names,
        segmentation_maps=segmentation_maps,
        segmentation_labels=build_segmentation_label_mapping(sim.mj_model),
        pos_atparent=np.array(pos_atparent_hist),
        pos_global=np.array(pos_global_hist),
        quat_global=np.array(quat_global_hist),
        body_segment_names=body_segment_names,
        cardinal_vectors=np.array(cardinal_vectors_hist),
        camera_matrix=np.array(camera_matrix_hist),
        fly_base_pos=np.array(fly_base_pos_hist),
    )

    if video_item is not None:
        top_frames, _ = build_period_frames(video_item, video_height)
        tracking_frames = tracking_renderer.frames[tracking_cam.name]
        # index 0 of CARDINAL_AXES/CARDINAL_LABELS is "x"/"forward".
        forward = np.array(cardinal_vectors_hist)[:, 0, :]
        yaw_deg = np.degrees(np.arctan2(forward[:, 1], forward[:, 0]))
        segid_rgb_dense_aligned = zoom_crop_frames(
            rotate_frames_to_heading(segid_colorized, yaw_deg),
            ZOOM_CROP_FRACTION,
            ZOOM_CROP_Y_OFFSET_FRACTION,
        )
        # The two pipelines' dense frame counts can differ by a frame or two
        # at period boundaries; keep only what all three have.
        n = min(len(top_frames), len(tracking_frames), len(segid_rgb_dense_aligned))
        combined_frames = [
            np.vstack(
                [
                    top_frames[i],
                    np.hstack([segid_rgb_dense_aligned[i], tracking_frames[i]]),
                ]
            )
            for i in range(n)
        ]
        write_video(
            combined_frames, output_dir / f"{output_stem}_combined.mp4", output_fps
        )

    logger.info(
        f"Saved {output_stem}: {len(sim_time_hist)} frames, "
        f"realtime factor {realtime_factor:.2f}x"
    )


# Cache for `_replay_one_period_in_worker`: one `build_fly_and_world()` result
# per joblib worker process, built lazily on that worker's first task. `fly`/
# `world`/the cameras hold live MuJoCo objects and can't be pickled across
# process boundaries, so (unlike `main`'s own single-process loop, which
# builds this once and reuses it directly) each worker process needs its own
# copy; caching it here avoids rebuilding it for every period that lands on
# an already-warm worker.
_worker_built = None


def _replay_one_period_in_worker(
    target_angles: np.ndarray,
    control_freq_hz: float,
    output_dir: Path,
    output_stem: str,
    video_height: int,
    playback_speed: float,
    video_item: dict | None,
) -> None:
    """`replay_period`, building/caching this worker's own fly/world first."""
    global _worker_built
    if _worker_built is None:
        _worker_built = build_fly_and_world()
    fly, world, tracking_cam, bottom_cam, bottom_cam_anchor, body_segments = (
        _worker_built
    )
    replay_period(
        fly,
        world,
        tracking_cam,
        bottom_cam,
        bottom_cam_anchor,
        body_segments,
        target_angles,
        control_freq_hz,
        output_dir,
        output_stem,
        video_height,
        playback_speed,
        video_item,
    )


def main(
    periods_path: tyro.conf.Positional[Path],
    output_dir: tyro.conf.Positional[Path],
    data_root: Path = DATA_ROOT,
    periods_per_trial: int = -1,
    max_trials: int | None = None,
    trial_name: str | None = None,
    n_replay_workers: int = 1,
    video_height: int = 450,
    playback_speed: float = 0.2,
    combine_panels: bool = True,
) -> None:
    """Replay retained IK periods in FlyGym (CPU) and render the result.

    Args:
        periods_path: IK/FK-augmented periods `.h5` (see `solve_ik.py`), with
            an `ik_dofangles_rad` dataset per period.
        output_dir: Directory to save the rendered replay videos and
            interoperable simulation data to. Cleared first if it already
            exists.
        data_root: Root directory containing each trial's `metadata.zip`
            (read-only), used to look up the trial's recording frame rate.
        periods_per_trial: Number of (longest) periods to replay per trial.
            `-1` replays every period in the trial.
        max_trials: If given, only replay the first N trials with a
            `metadata.zip` (in the periods `.h5`'s own iteration order), for
            quick testing. Unused if `trial_name` is given.
        trial_name: If given, restrict the replay to this one trial (e.g.
            `G-213xCI55_260720/fly000_trial000`, trailing slash optional),
            instead of every trial in `periods_path`.
        n_replay_workers: Number of periods replayed in parallel via joblib.
            Each worker builds its own `NeuroMechFly` model (MuJoCo objects
            aren't picklable across processes, so the model can't just be
            shared like it is in the `1`-worker case), reused for every
            period routed to it (see `_replay_one_period_in_worker`).
        video_height: Output height (and width; both cameras render square)
            in pixels, for every panel. The combined video (see
            `combine_panels`) is `2 * video_height` per side.
        playback_speed: Video playback speed relative to real time. The
            simulation itself already runs at the recording's own frame
            rate, so e.g. 0.2 plays the video back at 0.2x that real-world
            speed (slow motion), without changing which simulated frames get
            captured.
        combine_panels: If True, also build a 4-panel video per period (see
            `replay_period`), reusing `make_videos.py`'s 2D pose and
            synthetic 3D IK/FK panels for the top row. Skipped (with a
            warning) for periods whose aligned video is missing.
    """
    if not periods_path.is_file():
        raise SystemExit(f"Input file does not exist: {periods_path}")

    target_genotype_trial = None
    if trial_name is not None:
        parts = trial_name.strip("/").split("/")
        if len(parts) != 2:
            raise SystemExit(
                f"--trial-name must be '<genotype>/<fly_trial>', got {trial_name!r}"
            )
        target_genotype_trial = tuple(parts)

    # Only clear output_dir for a full (every-trial) run: with --trial-name,
    # this is expected to be one of many concurrent per-trial invocations
    # (e.g. a SLURM array job) sharing one output_dir, and clearing it here
    # would race with, and destroy, every other trial's already-written
    # output.
    if trial_name is None and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fly, world, tracking_cam, bottom_cam, bottom_cam_anchor, body_segments = (
        build_fly_and_world()
    )
    actuated_dof_order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    _, _, body_plan_dof_names = load_body_plan(BODY_PLAN_PATH)
    reorder_index = build_dof_reorder_index(body_plan_dof_names, actuated_dof_order)

    with h5py.File(periods_path, "r") as f:
        style = (
            compute_style_context(list(f.attrs["node_names"]))
            if combine_panels
            else None
        )

        work_items = []
        n_trials_processed = 0
        done = False
        for genotype in f:
            if done:
                break
            for fly_trial in f[genotype]:
                if target_genotype_trial is not None and (
                    genotype,
                    fly_trial,
                ) != target_genotype_trial:
                    continue
                if max_trials is not None and n_trials_processed >= max_trials:
                    done = True
                    break

                trial_dir = data_root / genotype / fly_trial
                if not (trial_dir / METADATA_ZIP_RELPATH).is_file():
                    logger.warning(
                        f"Skipping {genotype}/{fly_trial}: no metadata.zip at {trial_dir}"
                    )
                    continue
                n_trials_processed += 1
                control_freq_hz = load_control_freq_hz(trial_dir)

                video_path = trial_dir / ALIGNED_VIDEO_RELPATH
                if combine_panels and not video_path.is_file():
                    logger.warning(
                        f"{genotype}/{fly_trial}: no aligned video at {video_path}; "
                        "skipping combined-panel video for this trial."
                    )

                trial_group = f[genotype][fly_trial]
                period_ids = select_longest_periods(
                    trial_group, None if periods_per_trial == -1 else periods_per_trial
                )
                for period_id in period_ids:
                    group = trial_group[period_id]
                    if "ik_dofangles_rad" not in group:
                        raise SystemExit(
                            f"{periods_path} period {genotype}/{fly_trial}/"
                            f"{period_id} has no ik_dofangles_rad; run solve_ik.py first."
                        )
                    start_idx = int(group.attrs["start_idx"])
                    end_idx = int(group.attrs["end_idx"])
                    target_angles = group["ik_dofangles_rad"][:, reorder_index]
                    output_stem = f"{genotype}__{fly_trial}__period{period_id}_f{start_idx}-{end_idx}"

                    video_item = None
                    if combine_panels and video_path.is_file():
                        video_item = build_work_item(group, video_path, style, True)
                    work_items.append(
                        (target_angles, control_freq_hz, output_stem, video_item)
                    )
                logger.info(
                    f"{genotype}/{fly_trial}: queued {len(period_ids)} period replays "
                    f"(control_freq={control_freq_hz} Hz)"
                )

    if target_genotype_trial is not None and n_trials_processed == 0:
        raise SystemExit(f"No trial matching --trial-name={trial_name!r} found")

    logger.info(f"Replaying {len(work_items)} periods ({n_replay_workers} worker(s))")
    if n_replay_workers <= 1:
        for i, (target_angles, control_freq_hz, output_stem, video_item) in enumerate(
            work_items, start=1
        ):
            replay_period(
                fly,
                world,
                tracking_cam,
                bottom_cam,
                bottom_cam_anchor,
                body_segments,
                target_angles,
                control_freq_hz,
                output_dir,
                output_stem,
                video_height,
                playback_speed,
                video_item,
            )
            logger.info(f"Progress: {i}/{len(work_items)} periods done")
    else:
        Parallel(n_jobs=n_replay_workers, verbose=10)(
            delayed(_replay_one_period_in_worker)(
                target_angles,
                control_freq_hz,
                output_dir,
                output_stem,
                video_height,
                playback_speed,
                video_item,
            )
            for target_angles, control_freq_hz, output_stem, video_item in work_items
        )
    logger.info(f"Replayed {len(work_items)} periods to {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
