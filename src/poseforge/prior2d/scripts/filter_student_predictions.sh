#!/bin/bash -l
set -euo pipefail

scripts_dir="/home/sibwang/poseforge/src/poseforge/prior2d/scripts"
# The shared upstream .slp (LM model predictions, not score-threshold-specific)
# lives under the old lm_ported/ directory, kept as lm_ported_score_0.5/.
# Filtered/periods outputs go to the canonical lm_ported_score/ directory,
# named for the --min-keypoint-score value used below.
source_dir="/home/sibwang/poseforge/bulk_data/prior-2dinvkin/sleap/lm_ported_score_0.5"
output_dir="/home/sibwang/poseforge/bulk_data/prior-2dinvkin/sleap/lm_ported_score"

input_slp="$source_dir/lm_ported_v000_trained000.slp"
filtered_slp="$output_dir/filtered.slp"
filtered_h5="$output_dir/filtered.h5"
periods_h5="$output_dir/periods.h5"

[ -f "$input_slp" ] || { echo "File $input_slp does not exist"; exit 1; }
mkdir -p "$output_dir"

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

echo "Filtering finished at $(date). Converting filtered SLP to H5."
python "$scripts_dir/convert_slp.py" \
    --slp2h5 \
    --input-path "$filtered_slp" \
    --output-path "$filtered_h5" \
    --include-acceptance

echo "Conversion finished at $(date). Extracting continuous periods."
python "$scripts_dir/extract_continuous_periods_from_h5.py" \
    --input-path "$filtered_h5" \
    --output-path "$periods_h5"

echo "All done at $(date)."
