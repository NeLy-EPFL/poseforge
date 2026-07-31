#!/bin/bash -l
set -euo pipefail

# Runs the retrained single-instance model (trained on aligned-domain data,
# see README.md's "Round 1" caveat) directly on one trial's aligned video, no
# centroid stage needed since the model is single-instance. Unlike
# `run_lm_model_on_full_behavior_videos.sh` (the original two-stage LM
# model, run on the cluster on the raw fullsize video), this is meant to run
# locally against this repo checkout's own GPU.

trial_path="$1"
model_dir="/home/sibwang/Projects/poseforge/bulk_data/prior-2dinvkin/sleap/lm_ported_score_0.5/models/lmport_v000_run000.single_instance.n=467875"
scripts_dir="/home/sibwang/Projects/poseforge/src/poseforge/prior2d/scripts"

[ -d "$trial_path" ] || { echo "Directory $trial_path does not exist"; exit 1; }

cd "$trial_path"
mkdir -p "sleap"
video_path="processed/aligned_behavior_video.mkv"
slp_path="sleap/prediction_trained000_aligned.slp"
h5_path="sleap/prediction_trained000_aligned.h5"

echo "Running sleap-track at $(date)"
sleap-track \
    -m "$model_dir" \
    -o "$slp_path" \
    --batch_size 4 \
    "$video_path"

echo "sleap-track finished at $(date). Converting SLP to H5."
python "$scripts_dir/convert_slp.py" \
    --slp2h5 \
    --input-path "$slp_path" \
    --output-path "$h5_path"

echo "All done at $(date)."
