#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Usage check
# -----------------------------
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <config_path> <input_dir> <glob_pattern>"
    echo "Example: $0 config.yaml /data/videos \"*.mp4\""
    exit 1
fi

CONFIG_PATH="$1"
BASE_DIR="$2"
GLOB_PATTERN="$3"

ALIGNED_DIR="${BASE_DIR}/spotlight_aligned_and_cropped"
POSEFORGE_OUT_DIR="${BASE_DIR}/poseforge_output_tiled"

echo "Config path : ${CONFIG_PATH}"
echo "Base dir    : ${BASE_DIR}"
echo "Glob pattern: ${GLOB_PATTERN}"
echo "Aligned dir : ${ALIGNED_DIR}"
echo "PoseForge output dir : ${POSEFORGE_OUT_DIR}"

# -----------------------------
# Run pipeline
# -----------------------------

echo "Step 1: Splitting spotlight behavior videos into frames"
conda run -n poseforge-clone python \
    src/poseforge/spotlight/scripts/split_spotlight_behavior_video_to_frames.py \
    "${BASE_DIR}" \
    "${GLOB_PATTERN}"

echo "Step 2: Detecting flipped flies"
conda run -n poseforge-clone python \
    src/poseforge/spotlight/scripts/detect_flipped_flies.py \
    "${ALIGNED_DIR}" \
    "${GLOB_PATTERN}"\
    --config_path "${CONFIG_PATH}"

echo "Step 3: Running body segmentation inference"
conda run -n poseforge-clone python \
    src/poseforge/pose/bodyseg/scripts/run_bodyseg_inference.py \
    "${ALIGNED_DIR}" \
    "${GLOB_PATTERN}" \
    --config_path "${CONFIG_PATH}" \
    --output_basedir "${POSEFORGE_OUT_DIR}"

echo "Step 4: Running 3D keypoints inference"
conda run -n poseforge-clone python \
    src/poseforge/pose/keypoints3d/scripts/run_keypoints3d_inference.py \
    "${ALIGNED_DIR}" \
    "${GLOB_PATTERN}" \
    --config_path "${CONFIG_PATH}"\
    --output_basedir "${POSEFORGE_OUT_DIR}"


#echo "Step 5: Running inverse kinematics"
#conda run -n poseforge-clone python \
#    src/poseforge/pose/keypoints3d/scripts/run_inverse_kinematics.py \
#    "${ALIGNED_DIR}" \
#    "${GLOB_PATTERN}"\

echo
echo "Pipeline completed successfully."
