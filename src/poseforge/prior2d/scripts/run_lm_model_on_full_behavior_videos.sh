#!/bin/bash -l
set -euo pipefail

trial_path="$1"
centroid_model_dir="/home/sibwang/data/sleap/models/spotligh_lm/250327_002024.centroid"
centered_instance_model_dir="/home/sibwang/data/sleap/models/spotligh_lm/260701_180342.centered_instance.n=215"
scripts_dir="/home/sibwang/poseforge/src/poseforge/prior2d/scripts"

[ -d "$trial_path" ] || { echo "Directory $trial_path does not exist"; exit 1; }

cd "$trial_path"
mkdir -p "sleap"
video_path="processed/fullsize_behavior_video.mkv"
slp_path="sleap/prediction_lm_full_behavior_video.slp"
npz_path="sleap/prediction_lm_full_behavior_video.npz"
overview_video_path="sleap/prediction_lm_full_behavior_video.mp4"

echo "Running sleap-track at $(date)"
sleap-track \
    -m "$centroid_model_dir" \
    -m "$centered_instance_model_dir" \
    -o "$slp_path" \
    --batch_size 4 \
    "$video_path"

echo "sleap-track finished at $(date). Converting SLP to NPZ."
python "$scripts_dir/convert_slp.py" \
    --slp2npz \
    --input-path "$slp_path" \
    --output-path "$npz_path" \
    --video-path "$video_path" \
    --overview-video "$overview_video_path"

echo "All done at $(date)."
