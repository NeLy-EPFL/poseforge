import numpy as np
import pandas as pd
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

# Load simulation data
with h5py.File(sim_data_path, "r") as f:
    all_seg_pos_global = f["raw/body_segment_states/pos_global"][:]
    all_seg_quat_global = f["raw/body_segment_states/quat_global"][:]
    all_cam_matrices = f["raw/camera_matrix"][:]
    all_seg_names = list(f["raw/body_segment_states"].attrs["keys"])

n_frames = all_seg_pos_global.shape[0]
segments_to_include = [
    f"{leg}{seg}" for leg in legs for seg in all_segment_names_per_leg
]
segments_to_include += ["Thorax"]

# Load original meshes once (before any transformations)
original_meshes = {}
for seg_name in segments_to_include:
    mesh_file = nmf_mesh_dir / f"{seg_name}.stl"
    original_meshes[seg_name] = pv.read(mesh_file)

# Create plotter
plotter = pv.Plotter()
plotter.set_background("black")
plotter.show_axes()

# Add all meshes to plotter initially
current_meshes = {}
for seg_name in segments_to_include:
    current_meshes[seg_name] = original_meshes[seg_name].copy()
    plotter.add_mesh(
        current_meshes[seg_name], 
        show_edges=False, 
        name=seg_name, 
        smooth_shading=True
    )

# Current frame tracker
current_frame = [0]


def update_frame():
    """Update all meshes to the current frame"""
    frame_idx = current_frame[0]
    cam_matrix = all_cam_matrices[frame_idx, :, :]
    
    # Compute camera transformation once per frame
    cam_intrinsics, cam_rotation = rq(cam_matrix[:, :3])
    _sign_multiplier = np.diag(np.sign(np.diag(cam_intrinsics)))
    cam_intrinsics = cam_intrinsics @ _sign_multiplier
    cam_rotation = _sign_multiplier @ cam_rotation
    cam_translation = np.linalg.inv(cam_intrinsics) @ cam_matrix[:, 3]
    
    transform_world2cam = np.eye(4)
    transform_world2cam[:3, :3] = cam_rotation
    transform_world2cam[:3, 3] = cam_translation
    transform_cam2world = np.linalg.inv(transform_world2cam)
    
    # Update each segment mesh
    for seg_name in segments_to_include:
        seg_idx = all_seg_names.index(seg_name)
        translation = all_seg_pos_global[frame_idx, seg_idx, :]
        quaternion = all_seg_quat_global[frame_idx, seg_idx, :]
        
        # Start with original mesh
        mesh = original_meshes[seg_name].copy()
        
        # Scale
        scale_transform = np.eye(4)
        np.fill_diagonal(scale_transform, [1000, 1000, 1000, 1])
        mesh = mesh.transform(scale_transform, inplace=False)
        
        # Apply MuJoCo state transformation
        placement_transform = np.eye(4)
        rotation_object = Rotation.from_quat(quaternion, scalar_first=True)
        placement_transform[:3, :3] = rotation_object.as_matrix()
        placement_transform[:3, 3] = translation
        mesh = mesh.transform(placement_transform, inplace=False)
        
        # Transform to camera coordinates
        mesh = mesh.transform(transform_cam2world, inplace=False)
        
        # Update the mesh points in place
        current_meshes[seg_name].points[:] = mesh.points
    
    # Update frame counter
    current_frame[0] = (current_frame[0] + 1) % n_frames
    
    # Update title to show current frame
    plotter.add_text(f"Frame: {frame_idx}/{n_frames}", name="frame_counter", position="upper_left")


# Initialize first frame
update_frame()
plotter.reset_camera()

# Add timer callback for animation (30 fps)
# The callback receives a step argument, so we need to accept it
plotter.add_timer_event(max_steps=n_frames, duration=int(1000/30), callback=lambda step: update_frame())

plotter.show()