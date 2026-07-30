#!/bin/bash -l
set -euo pipefail

scripts_dir="/home/sibwang/poseforge/src/poseforge/prior2d/scripts"
lm_ported_dir="/home/sibwang/poseforge/bulk_data/prior-2dinvkin/sleap/lm_ported"

input_slp="$lm_ported_dir/lm_ported_v000_trained000.slp"
filtered_slp="$lm_ported_dir/lm_ported_v000_trained000_filtered.slp"
filtered_npz="$lm_ported_dir/lm_ported_v000_trained000_filtered.npz"
periods_h5="$lm_ported_dir/lm_ported_v000_trained000_filtered_periods.h5"

[ -f "$input_slp" ] || { echo "File $input_slp does not exist"; exit 1; }

# Same selection criteria as port_slp_labels.py + filter_slp_labels.py's
# defaults, spelled out here so this run doesn't silently drift if those
# defaults ever change.
echo "Filtering student model predictions at $(date)"
python "$scripts_dir/filter_slp_labels.py" \
    --input-path "$input_slp" \
    --output-path "$filtered_slp" \
    --aligned-input \
    --min-keypoint-score 0.2 \
    --max-leg-segment-length 200.0 \
    --max-missing-keypoints 0 \
    --min-pose-change 2.0

echo "Filtering finished at $(date). Converting filtered SLP to NPZ."
python "$scripts_dir/convert_slp.py" \
    --slp2npz \
    --input-path "$filtered_slp" \
    --output-path "$filtered_npz" \
    --include-acceptance

echo "Conversion finished at $(date). Extracting continuous periods."
python "$scripts_dir/extract_continuous_periods_from_npz.py" \
    --input-path "$filtered_npz" \
    --output-path "$periods_h5"

echo "All done at $(date)."
