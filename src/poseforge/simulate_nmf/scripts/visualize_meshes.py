import numpy as np
import pandas as pd
import pyvista as pv
import flygym
from xml.etree import ElementTree
from pathlib import Path
from scipy.spatial.transform import Rotation


df = pd.read_pickle(
    "bulk_data/nmf_rendering_enhanced/BO_Gal4_fly1_trial001/segment_000/subsegment_000/processed_kinematic_states.pkl"
)
flygym_data_dir = Path(flygym.__file__).parent / "data"
nmf_mesh_dir = flygym_data_dir / "mesh"
mjcf_path = flygym_data_dir / "mjcf/neuromechfly_seqik_kinorder_ypr.xml"

sides = "LR"
positions = "FMH"
links = [
    "Coxa",
    "Femur",
    "Tibia",
    "Tarsus1",
    "Tarsus2",
    "Tarsus3",
    "Tarsus4",
    "Tarsus5",
]

mjcf_tree = ElementTree.parse(mjcf_path)
worldbody = mjcf_tree.find("worldbody")
body_attributes = {body.attrib["name"]: body.attrib for body in worldbody.iter("body")}
frame_idx = 10
entry = df.loc[frame_idx]


plotter = pv.Plotter()

segments_to_include = [
    f"{side}{pos}{link}" for side in sides for pos in positions for link in links
]
segments_to_include += ["Thorax"]
meshes = []
for key in segments_to_include:
    translation = entry[f"body_seg_pos_global_{key}"]
    quaternion = entry[f"body_seg_quat_global_{key}"]
    placement_transform = np.eye(4)
    placement_transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    placement_transform[:3, 3] = translation

    mesh_file = nmf_mesh_dir / f"{key}.stl"
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

    meshes.append(mesh)
    plotter.add_mesh(mesh, show_edges=False, name=key, smooth_shading=True)

    # print(f"Loading {key}:\n\ttranslation={translation}\n\tquaternion={quaternion}")


plotter.set_background("black")
plotter.reset_camera()
plotter.show_axes()
plotter.show()  # or plotter.show(screenshot='screenshot.png')
