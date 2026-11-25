#!/bin/bash
# Submit all four conditions: {randomcrop, curriculum} x {imageonly, allpriors}
set -euo pipefail
cd "$(dirname "$0")"

for REGIME in randomcrop curriculum; do
    for BUFFERS in imageonly allpriors; do
        echo "Submitting mp_${REGIME}_${BUFFERS}"
        sbatch --job-name "mp_${REGIME}_${BUFFERS}" job.run "$REGIME" "$BUFFERS"
    done
done
