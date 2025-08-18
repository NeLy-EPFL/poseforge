"""Load data from F. Aymanns et al. (2022) from lab server (also available
on Harvard Dataverse: https://doi.org/10.7910/DVN/QQMNQK). This includes
two files for each trial: one containing neural network classification of
the behavior type for each frame of the video, and one containing the
neural network prediction of kinematic state (joint angles, etc.). This
script reads from both files, merge them, and write the result to a single
.pkl file in the project data directory.

This script is reused from the variational VNC modeling project:
https://github.com/NeLy-EPFL/vvnc
"""

import json
import pandas as pd
from pathlib import Path


# Parameters
source_basedir = Path(
    "/mnt/upramdya_archives/common/_NAS1_data2/FA/data_for_Aymanns_et_al/dataverse/"
)
target_dir = Path("bulk_data/kinematic_recording/aymanns2022/trials")
expected_fps: int = 100


# Index what data we have
datasets = sorted(list(source_basedir.glob("*/Fly*/*_trial*")))
datasets = [x for x in datasets if x.suffix != ".zip"]
print(f"Found {len(datasets)} datasets.")

# Copy data
target_dir.mkdir(parents=True, exist_ok=True)
print(f"Target directory: {target_dir}")
for dataset_dir in datasets:
    fly_and_trial = dataset_dir.name
    genotype = dataset_dir.parent.parent.name
    experiment_label = genotype + "_" + fly_and_trial
    capture_metadata_file = dataset_dir / "behData/images/capture_metadata.json"
    beh_classification_file = (
        dataset_dir / "behData/images/df3d/behaviour_predictions_daart.pkl"
    )
    beh_kinematics_file = dataset_dir / "behData/images/df3d/post_processed.pkl"
    print(f"Loading, merging, and writing data for {experiment_label}")

    # Make sure the FPS is what we expect
    with open(capture_metadata_file, "r") as f:
        metadata = json.load(f)
    if metadata["FPS"] != expected_fps:
        print(
            "Skipping: unexpected FPS indicated in "
            f"capture_metadata.json ({metadata['FPS']})"
        )
        continue

    # Load and merge data
    beh_class = pd.read_pickle(beh_classification_file)
    beh_kin = pd.read_pickle(beh_kinematics_file)
    merged_df = pd.merge(beh_class, beh_kin, left_index=True, right_index=True)
    merged_df.to_pickle(target_dir / f"{experiment_label}.pkl")

print("Done.")
