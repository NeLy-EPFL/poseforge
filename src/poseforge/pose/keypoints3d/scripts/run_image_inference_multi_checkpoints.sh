#!/usr/bin/env bash
set -euo pipefail

# Set this to 1 to use the hardcoded list below by default.
USE_HARDCODED_CHECKPOINTS=1

# Edit this list with the checkpoints you want to run.
HARDCODED_CHECKPOINTS=(
    "/Volumes/upramdya/data/VAS/poseforge/production/pose_estimation/keypoints3d/trial_20260507_tiled/checkpoints/epoch19_step2039.model.pth"
    "/Volumes/upramdya/data/VAS/poseforge/production/pose_estimation/keypoints3d/trial_20260507_tiled_01noise/checkpoints/epoch28_step2039.model.pth"
    "/Volumes/upramdya/data/VAS/poseforge/production/pose_estimation/keypoints3d/trial_20260507_tiled_05noise/checkpoints/epoch28_step2039.model.pth"
)

if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 <input_path> [<checkpoint_1> ... <checkpoint_n>] [-- <extra_run_image_inference_args>]" >&2
    echo "Examples:" >&2
    echo "  # Use hardcoded checkpoints" >&2
    echo "  $0 /path/to/images -- --n_images 8 --batch_size 2" >&2
    echo "  # Override with explicit checkpoints" >&2
    echo "  $0 /path/to/images /path/to/a.model.pth /path/to/b.model.pth -- --n_images 8 --batch_size 2" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/run_image_inference.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "Inference script not found: $PY_SCRIPT" >&2
    exit 1
fi

input_path="$1"
shift

if [[ ! -e "$input_path" ]]; then
    echo "Input path not found: $input_path" >&2
    exit 1
fi

cli_checkpoints=()
checkpoints=()
extra_args=()
seen_separator=0

for arg in "$@"; do
    if [[ "$arg" == "--" && "$seen_separator" -eq 0 ]]; then
        seen_separator=1
        continue
    fi

    if [[ "$seen_separator" -eq 0 ]]; then
        cli_checkpoints+=("$arg")
    else
        extra_args+=("$arg")
    fi
done

if [[ "${#cli_checkpoints[@]}" -gt 0 ]]; then
    checkpoints=("${cli_checkpoints[@]}")
elif [[ "$USE_HARDCODED_CHECKPOINTS" -eq 1 ]]; then
    checkpoints=("${HARDCODED_CHECKPOINTS[@]}")
else
    echo "No checkpoints provided. Either pass checkpoints on CLI or set HARDCODED_CHECKPOINTS in this script." >&2
    exit 1
fi

if [[ "${#checkpoints[@]}" -eq 0 ]]; then
    echo "Checkpoint list is empty. Fill HARDCODED_CHECKPOINTS or pass checkpoints via CLI." >&2
    exit 1
fi

for ckpt in "${checkpoints[@]}"; do
    if [[ ! -f "$ckpt" ]]; then
        echo "Checkpoint not found: $ckpt" >&2
        exit 1
    fi

done

for ckpt in "${checkpoints[@]}"; do
    echo "Running inference with checkpoint: $ckpt"
    python "$PY_SCRIPT" "$input_path" --checkpoint "$ckpt" "${extra_args[@]}"
done

echo "Completed inference for ${#checkpoints[@]} checkpoint(s)."
