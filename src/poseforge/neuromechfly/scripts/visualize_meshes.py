import numpy as np
import pyvista as pv
import h5py
from xml.etree import ElementTree
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.linalg import rq

from poseforge.neuromechfly.constants import legs, all_segment_names_per_leg


# Define paths
subsegment_dir = Path(
    "bulk_data/nmf_rendering/BO_Gal4_fly1_trial001/segment_000/subsegment_000"
)
sim_data_path = subsegment_dir / "processed_simulation_data.h5"
flygym_data_dir = Path("~/projects/flygym/flygym").expanduser() / "data"
nmf_mesh_dir = flygym_data_dir / "mesh"
mjcf_path = flygym_data_dir / "mjcf/neuromechfly_seqik_kinorder_ypr.xml"

# Load NeuroMechFly model
mjcf_tree = ElementTree.parse(mjcf_path)
worldbody = mjcf_tree.find("worldbody")
body_attributes = {body.attrib["name"]: body.attrib for body in worldbody.iter("body")}
frame_idx = 10

# Load simulation data
with h5py.File(sim_data_path, "r") as f:
    # all_seg_pos_global: (n_frames, n_segments, 3)
    all_seg_pos_global = f["raw/body_segment_states/pos_global"][:]
    # all_seg_quat_global: (n_frames, n_segments, 4)
    all_seg_quat_global = f["raw/body_segment_states/quat_global"][:]
    # all_cam_matrices: (n_frames, 3, 4)
    all_cam_matrices = f["raw/camera_matrix"][:]
    # all_seg_names: list of length n_segments
    all_seg_names = list(f["raw/body_segment_states"].attrs["keys"])

plotter = pv.Plotter()

segments_to_include = [
    f"{leg}{seg}" for leg in legs for seg in all_segment_names_per_leg
]
segments_to_include += ["Thorax"]
meshes = []
for seg_name in segments_to_include:
    seg_idx = all_seg_names.index(seg_name)
    translation = all_seg_pos_global[frame_idx, seg_idx, :]
    quaternion = all_seg_quat_global[frame_idx, seg_idx, :]
    cam_matrix = all_cam_matrices[frame_idx, :, :]

    placement_transform = np.eye(4)
    placement_transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    placement_transform[:3, 3] = translation

    mesh_file = nmf_mesh_dir / f"{seg_name}.stl"
    mesh = pv.read(mesh_file)

    # Scale
    scale_transform = np.eye(4)
    np.fill_diagonal(scale_transform, [1000, 1000, 1000, 1])
    mesh.transform(scale_transform)

    # Apply transformation based on MuJoCo state
    placement_transform = np.eye(4)
    rotation_object = Rotation.from_quat(quaternion, scalar_first=True)
    placement_transform[:3, :3] = rotation_object.as_matrix()
    placement_transform[:3, 3] = translation
    mesh.transform(placement_transform)

    # Convert to coordinates relative to camera
    # Given camera projection matrix P = cam_matrix (3x4), solve for:
    #   R = camera rotation matrix (3x3)
    #   K = camera intrinsic matrix (3x3)
    # where P = K[R|t]
    # First, run RQ decomposition to get K and R
    cam_intrinsics, cam_rotation = rq(cam_matrix[:, :3])
    # Make camera intrinsic mastrix have positive diagonal (just a convention)
    _sign_multiplier = np.diag(np.sign(np.diag(cam_intrinsics)))
    cam_intrinsics = cam_intrinsics @ _sign_multiplier
    cam_rotation = _sign_multiplier @ cam_rotation
    # Get camera translation in world coords
    cam_translation = np.linalg.inv(cam_intrinsics) @ cam_matrix[:, 3]

    transform_world2cam = np.eye(4)
    transform_world2cam[:3, :3] = cam_rotation
    transform_world2cam[:3, 3] = cam_translation
    # print(transform_world2cam)
    # Cam matrix describes cam state in world coords: invert it for cam-to-world mapping
    transform_cam2world = np.linalg.inv(transform_world2cam)
    mesh.transform(transform_cam2world)

    meshes.append(mesh)
    plotter.add_mesh(mesh, show_edges=False, name=seg_name, smooth_shading=True)


plotter.set_background("black")
plotter.reset_camera()
plotter.show_axes()
plotter.show()  # or plotter.show(screenshot='screenshot.png')
