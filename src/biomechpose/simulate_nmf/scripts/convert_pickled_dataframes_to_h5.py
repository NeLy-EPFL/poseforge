"""
Data from NeuroMechFly simulations were initially saved as pickled pandas
DataFrames. This turned out to be a poor choice because some columns
contain numpy ndarrays. This makes IO very slow (up to 800ms for opening a
~20MB with ~60 ndarray columns). This script converts the pickled
DataFrames to HDF5 files. The simulation code has been modified to save
HDF5 files directly, so this script only needs to be run once on existing
data (which unfortunately is the entire Aymanns et al. 2022 dataset).

Note that data is converted at the segment level, i.e. the raw outputs of
each NeuroMechFly simulation before any postprocessing. The postprocessing
code has been modified to take these HDF5 files as input directly, so I
simply reran the updated postprocessing code on the outputs of this script.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def convert_simulated_segment_data_to_h5(sim_results_dir: Path):
    h5_path = sim_results_dir / "simulation_data.h5"
    kinematics_path = sim_results_dir / "kinematic_states_history.pkl"

    if not kinematics_path.is_file():
        return  # this simulation crashed early and has no usable output

    with h5py.File(h5_path, "w") as h5_file:
        convert_kinematics_dataframe_to_h5(h5_file, kinematics_path)


def convert_kinematics_dataframe_to_h5(
    h5_file: h5py.File, kinematics_path: Path
) -> None:
    pd_df = pd.read_pickle(kinematics_path)
    n_timesteps = len(pd_df)
    h5_file.attrs["n_timesteps"] = n_timesteps
    _n_processed_columns = 0

    # Time
    time_ds = h5_file.create_dataset(
        "sim_time", data=pd_df["time"].to_numpy(), dtype="float32"
    )
    time_ds.attrs["units"] = "s"
    time_ds.attrs["description"] = "Time in the NeuroMechFly simulation"
    _n_processed_columns += 1

    # DoF angles
    dof_columns = [col for col in pd_df.columns if col.startswith("dof_angle_")]
    dof_angles_ds = h5_file.create_dataset(
        "joint_angles", data=pd_df[dof_columns].to_numpy(), dtype="float32"
    )
    dof_angles_ds.attrs["keys"] = [col[len("dof_angle_") :] for col in dof_columns]
    dof_angles_ds.attrs["units"] = "radians"
    dof_angles_ds.attrs["description"] = (
        "Angles of DoFs tracked in the simulation. "
        "This dataset has shape (n_timesteps, n_dofs). The order of the DoFs is given "
        "in the 'order' attribute."
    )
    _n_processed_columns += len(dof_columns)

    # Body segment state
    body_seg_group = h5_file.create_group("body_segment_states")
    segments_order = None
    for pos_or_quat in ["pos", "quat"]:
        for ref_frame in ["atparent", "com", "global"]:
            # Check if this variable is saved
            prefix = f"body_seg_{pos_or_quat}_{ref_frame}_"
            columns = [col for col in pd_df.columns if col.startswith(prefix)]
            if not columns:
                continue

            # Check if body segment order is consistent
            my_columns_order = [col[len(prefix) :] for col in columns]
            if segments_order is None:
                segments_order = my_columns_order
            else:
                assert (
                    segments_order == my_columns_order
                ), "Inconsistent body segment order"

            # Create h5 subdataset
            n_dim = 3 if pos_or_quat == "pos" else 4
            data_block = np.empty(
                (n_timesteps, len(segments_order), n_dim), dtype="float32"
            )
            for i in range(n_timesteps):
                for j in range(len(segments_order)):
                    data_block[i, j, :] = pd_df[columns[j]].iloc[i]
            this_ds = body_seg_group.create_dataset(
                f"{pos_or_quat}_{ref_frame}", data=data_block, dtype="float32"
            )
            if pos_or_quat == "pos":
                this_ds.attrs["keys"] = ["x", "y", "z"]
                this_ds.attrs["units"] = "mm"
                this_ds.attrs["description"] = (
                    f'Position of each body segment in the "{ref_frame}" reference '
                    "frame. This dataset has shape (n_timesteps, n_segments, 3). The "
                    "order of the segments is given in the 'order' attribute."
                )
            else:
                this_ds.attrs["keys"] = ["w", "x", "y", "z"]
                this_ds.attrs["units"] = "quaternion"
                this_ds.attrs["description"] = (
                    "Orientation (as a quaternion) of each body segment in the "
                    f'"{ref_frame}" reference frame. This dataset has shape '
                    "(n_timesteps, n_segments, 4). The order of the segments is given "
                    "in the 'order' attribute."
                )
            _n_processed_columns += len(columns)

    body_seg_group.attrs["keys"] = segments_order
    body_seg_group.attrs["description"] = (
        "Position (in mm) and orientation (as quaternions) of each body segment "
        "tracked in the simulation. Values are provided in several reference frames: "
        "'atparent' corresponds to 'xbody' in MuJoCo: 'the regular frame of the body "
        "(usually centered at the joint with the parent body)'; `com` corresponds to "
        "'body' in MuJoCo: 'the inertial frame of the body'; `global` is the position "
        "and orientation in the global/world reference frame. See "
        "https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-framepos "
        "for the distinction between these `body` and `xbody`."
    )

    # Cardinal vectors
    cardinal_vec_group = h5_file.create_group("cardinal_vectors")
    cardinal_vec_group.attrs["keys"] = ["forward", "left", "up"]
    cardinal_vec_group.attrs["description"] = (
        "Unit vectors pointing in cardinal directions from the perspective of the fly, "
        "i.e. vector pointing forward, to the left, and up from the fly's body. "
        "These vectors are in global coordinates and each of them is of shape "
        "(n_timesteps, 3) where 3 are the x/y/z components."
    )
    for vec_direction in cardinal_vec_group.attrs["keys"]:
        data_block = np.vstack(pd_df[f"cardinal_vector_{vec_direction}"])
        this_ds = cardinal_vec_group.create_dataset(
            vec_direction, data=data_block, dtype="float32"
        )
        this_ds.attrs["keys"] = ["x", "y", "z"]
        _n_processed_columns += 1

    # Camera matrix
    cam_mat_ds = h5_file.create_dataset(
        "camera_matrix", data=np.stack(pd_df["camera_matrix"]), dtype="float32"
    )
    cam_mat_ds.attrs["description"] = (
        "3x4 camera matrix (numpy array) for each frame "
        "(see https://en.wikipedia.org/wiki/Camera_matrix)."
    )
    _n_processed_columns += 1

    # Fly base position
    fly_base_pos_ds = h5_file.create_dataset(
        "fly_base_pos", data=np.vstack(pd_df["fly_base_pos"]), dtype="float32"
    )
    fly_base_pos_ds.attrs["keys"] = ["x", "y", "z"]
    fly_base_pos_ds.attrs["units"] = "mm"
    fly_base_pos_ds.attrs["description"] = (
        "Position of the fly's center of mass in global coordinates. This dataset has "
        "shape (n_timesteps, 3) where the 3 values are the x/y/z components."
    )
    _n_processed_columns += 1

    assert _n_processed_columns == len(
        pd_df.columns
    ), "Number of processed columns does not match number of columns in the DataFrame."


if __name__ == "__main__":
    from joblib import Parallel, delayed

    # Process all simulations at the segment level
    sim_output_basedir = Path("bulk_data/nmf_rendering")
    all_sim_dirs = sorted(
        [path for path in sim_output_basedir.rglob("segment_*") if path.is_dir()]
    )

    # # Process in series (for testing)
    # for sim_dir in tqdm(all_sim_dirs, disable=None):
    #     convert_simulated_segment_data_to_h5(sim_dir)

    # Process in parallel (5 minutes for 1450 segments in 16 threads)
    Parallel(n_jobs=-1)(
        delayed(convert_simulated_segment_data_to_h5)(sim_dir)
        for sim_dir in tqdm(all_sim_dirs, disable=None)
    )
