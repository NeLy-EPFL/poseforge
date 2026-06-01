#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Run video inference for multiple checkpoints.
#
# Usage:
#   ./run_video_inference_multi_checkpoints.sh <video_path> [-- extra_args]
#
# Checkpoints are defined in the CHECKPOINTS array below. Edit it to add
# or remove checkpoints. Each produces a separate output video named:
#   <video_stem>_<run_name>_<checkpoint_stem>.mp4
#
# Examples:
#   # Run all checkpoints with defaults
#   ./run_video_inference_multi_checkpoints.sh /path/to/video.mp4
#
#   # Pass extra args (batch size, device, etc.)
#   ./run_video_inference_multi_checkpoints.sh /path/to/video.mp4 -- --batch_size 32 --device cpu
# ============================================================================

# ---- Edit this list with your checkpoints ----
CHECKPOINTS=(
    "/home/stimpfling/poseforge/production_models/keypoints3d/checkpoints/epoch19_step9167.model.pth"
    "/mnt/upramdya_data/VAS/poseforge/production/pose_estimation/keypoints3d/trial_20260507_tiled/checkpoints/epoch19_step2039.model.pth"
    "/mnt/upramdya_data/VAS/poseforge/improve_kpt/trial_20260529_tiled_keypoints3d_drop03/checkpoints/epoch6_step1439.model.pth"
    "/mnt/upramdya_data/VAS/poseforge/improve_kpt/trial_20260529_tiled_keypoints3d_freeze/checkpoints/epoch17_step1000.model.pth"
    "/mnt/upramdya_data/VAS/poseforge/improve_kpt/trial_20260529_tiled_keypoints3d_hs8/checkpoints/epoch15_step1439.model.pth"
    "/mnt/upramdya_data/VAS/poseforge/improve_kpt/trial_20260529_tiled_keypoints3d_nodepth/checkpoints/epoch14_step1000.model.pth"
    # Add more checkpoints here, one per line:
    # "/path/to/another/trial_.../checkpoints/epochN_stepM.model.pth"
)

# ---- Parse arguments ----
if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 <video_path> [-- <extra_args>]" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  $0 /path/to/video.mp4" >&2
    echo "  $0 /path/to/video.mp4 -- --batch_size 32 --device cpu --conf_threshold 0.3" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/run_video_inference.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "Python inference script not found: $PY_SCRIPT" >&2
    exit 1
fi

video_path="$1"
shift

if [[ ! -f "$video_path" ]]; then
    echo "Video file not found: $video_path" >&2
    exit 1
fi

# Collect extra args after "--"
extra_args=()
seen_separator=0
for arg in "$@"; do
    if [[ "$arg" == "--" && "$seen_separator" -eq 0 ]]; then
        seen_separator=1
        continue
    fi
    if [[ "$seen_separator" -eq 1 ]]; then
        extra_args+=("$arg")
    fi
done

# ---- Validate checkpoints ----
if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
    echo "No checkpoints defined. Edit the CHECKPOINTS array in this script." >&2
    exit 1
fi

for ckpt in "${CHECKPOINTS[@]}"; do
    if [[ ! -f "$ckpt" ]]; then
        echo "WARNING: Checkpoint not found (will skip): $ckpt" >&2
    fi
done

# ---- Run inference for each checkpoint ----
echo "======================================"
echo "Video: $video_path"
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "Extra args: ${extra_args[*]:-<none>}"
echo "======================================"

n_done=0
n_skipped=0

for ckpt in "${CHECKPOINTS[@]}"; do
    if [[ ! -f "$ckpt" ]]; then
        echo "SKIP (not found): $ckpt"
        ((n_skipped++)) || true
        continue
    fi

    # Extract run name for display
    run_dir="$(dirname "$(dirname "$ckpt")")"
    run_name="$(basename "$run_dir")"
    ckpt_name="$(basename "$ckpt")"

    echo ""
    echo "--- [$((n_done + 1))/${#CHECKPOINTS[@]}] $run_name / $ckpt_name ---"
    python "$PY_SCRIPT" "$video_path" --checkpoint "$ckpt" "${extra_args[@]}"
    ((n_done++)) || true
done

echo ""
echo "======================================"
echo "Completed: $n_done / ${#CHECKPOINTS[@]} checkpoints ($n_skipped skipped)"
echo "======================================"
