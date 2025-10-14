import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from poseforge.pose_estimation.keypoints3d.invkin import (
    run_seqikpy,
    save_seqikpy_output,
)


def process_all(
    input_dirs: list[str],
    max_n_frames: int | None = None,
    n_workers_per_dataset: int = 6,
) -> None:
    # Index all keypoints3d output files to process
    all_keypoints3d_output_files = []
    for input_dir in input_dirs:
        keypoints3d_output_files = list(Path(input_dir).rglob("keypoints3d.h5"))
        all_keypoints3d_output_files.extend(keypoints3d_output_files)
    all_keypoints3d_output_files = sorted(all_keypoints3d_output_files)

    for keypoints3d_output_file in tqdm(all_keypoints3d_output_files):
        with h5py.File(keypoints3d_output_file, "r") as f:
            frame_ids = f["frame_ids"][:]
            world_xyz = f["keypoints_world_xyz"][:]
            keypoint_names_canonical = f["keypoints_world_xyz"].attrs["keypoints"]
        output_path = keypoints3d_output_file.parent / "inverse_kinematics.h5"
        joint_angles, forward_kinematics = run_seqikpy(
            world_xyz=world_xyz,
            keypoint_names_canonical=keypoint_names_canonical,
            max_n_frames=max_n_frames,
            n_workers=n_workers_per_dataset,
        )
        save_seqikpy_output(
            output_path, joint_angles, forward_kinematics, frame_ids=frame_ids
        )


if __name__ == "__main__":
    import tyro

    tyro.cli(
        process_all,
        prog=f"python {Path(__file__).name}",
        description="Run inverse kinematics on all keypoints3d output files in the given directories.",
    )
