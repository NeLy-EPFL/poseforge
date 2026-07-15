#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Usage check
# -----------------------------
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <config_path> <input_dir> <glob_pattern> [n_workers]"
    echo "Example: $0 config.yaml /data/videos \"fly*\" 16"
    exit 1
fi

CONFIG_PATH="$1"
BASE_DIR="$2"
GLOB_PATTERN="$3"
# Number of CPU workers for data loading / frame extraction.
N_WORKERS="${4:-16}"

ALIGNED_DIR="${BASE_DIR}/spotlight_aligned_and_cropped"
POSEFORGE_OUT_DIR="${BASE_DIR}/poseforge_output_tiled"

# Live, unbuffered stdout so progress bars stream through `conda run`.
export PYTHONUNBUFFERED=1

echo "Config path : ${CONFIG_PATH}"
echo "Base dir    : ${BASE_DIR}"
echo "Glob pattern: ${GLOB_PATTERN}"
echo "Aligned dir : ${ALIGNED_DIR}"
echo "PoseForge output dir : ${POSEFORGE_OUT_DIR}"
echo "Workers     : ${N_WORKERS}"

# Report the GPU the CUDA steps will use (non-fatal if nvidia-smi is absent).
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    | sed 's/^/GPU         : /' || echo "GPU         : (nvidia-smi not found)"

# -----------------------------
# Run pipeline
# -----------------------------

echo "Step 1: Splitting spotlight behavior videos into frames (parallel, CPU)"
conda run -n poseforge-clone python \
    src/poseforge/spotlight/scripts/split_spotlight_behavior_video_to_frames.py \
    "${BASE_DIR}" \
    "${GLOB_PATTERN}" \
    --n_workers "${N_WORKERS}"

echo "Step 2: Detecting flipped flies (CUDA, batched)"
# Skip trials that already have flip predictions; run detection only on the
# rest. Pass FORCE_FLIP=1 to re-run every trial regardless.
for trial_dir in "${ALIGNED_DIR}"/${GLOB_PATTERN}/; do
    [ -d "${trial_dir}" ] || continue
    trial_name="$(basename "${trial_dir}")"
    if [ "${FORCE_FLIP:-0}" != "1" ] && [ -f "${trial_dir}predicted_flip_labels.csv" ]; then
        echo "  Skipping ${trial_name} (predicted_flip_labels.csv exists)"
        continue
    fi
    echo "  Detecting flips: ${trial_name}"
    conda run -n poseforge-clone python \
        src/poseforge/spotlight/scripts/detect_flipped_flies.py \
        "${ALIGNED_DIR}" \
        "${trial_name}" \
        --config_path "${CONFIG_PATH}" \
        --device cuda \
        --n_workers "${N_WORKERS}"
done

echo "Step 3: Running body segmentation inference (CUDA, batched)"
conda run -n poseforge-clone python \
    src/poseforge/pose/bodyseg/scripts/run_bodyseg_inference.py \
    "${ALIGNED_DIR}" \
    "${GLOB_PATTERN}" \
    --config_path "${CONFIG_PATH}" \
    --output_basedir "${POSEFORGE_OUT_DIR}"

echo "Step 4: Running 3D keypoints inference (CUDA, batched)"
conda run -n poseforge-clone python \
    src/poseforge/pose/keypoints3d/scripts/run_keypoints3d_inference.py \
    "${ALIGNED_DIR}" \
    "${GLOB_PATTERN}" \
    --config_path "${CONFIG_PATH}" \
    --output_basedir "${POSEFORGE_OUT_DIR}"


#echo "Step 5: Running inverse kinematics"
#conda run -n poseforge-clone python \
#    src/poseforge/pose/keypoints3d/scripts/run_inverse_kinematics.py \
#    "${ALIGNED_DIR}" \
#    "${GLOB_PATTERN}"\

echo
echo "Pipeline completed successfully."
