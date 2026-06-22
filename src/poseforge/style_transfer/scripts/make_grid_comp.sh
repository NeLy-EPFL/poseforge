#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="${1:-.}"

find "$BASE_DIR" -type f -path "*/validation/epoch*.mp4" | while read -r EPOCH_FILE; do

    DIR=$(dirname "$EPOCH_FILE")
    BASE_NAME=$(basename "$EPOCH_FILE")

    # Extract key: epoch392_examplesim00 -> examplesim00
    KEY=$(echo "$BASE_NAME" | sed -E 's/epoch[0-9]+_//; s/\.mp4//')

    REF_FILE="${DIR}/input_reference_${KEY}.mp4"

    if [[ ! -f "$REF_FILE" ]]; then
        echo "⚠️ Missing reference for $EPOCH_FILE"
        continue
    fi

    OUTPUT_DIR="${DIR}/grid_comps"
    mkdir -p "$OUTPUT_DIR"
    OUT_FILE="${OUTPUT_DIR}/${BASE_NAME}"

    echo "🎬 Processing:"
    echo "  LEFT (reference): $REF_FILE"
    echo "  RIGHT (epoch):    $EPOCH_FILE"
    echo "  OUT:              $OUT_FILE"

    ffmpeg -y \
        -i "$REF_FILE" \
        -i "$EPOCH_FILE" \
        -filter_complex "\
        [0:v]drawgrid=w=iw/5:h=ih/5:t=1:c=red@0.3[ref]; \
        [1:v]drawgrid=w=iw/5:h=ih/5:t=1:c=red@0.3[epoch]; \
        [ref][epoch]hstack=inputs=2" \
        -c:v libx264 -preset fast -crf 18 \
        -an \
        "$OUT_FILE"

done

echo "✅ Done. Output saved to: $OUTPUT_DIR"