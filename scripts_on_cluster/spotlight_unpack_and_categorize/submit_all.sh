#!/bin/bash
# Submit one unpack+categorize job per genotype.
set -euo pipefail
cd "$(dirname "$0")"

GENOTYPES=(G1xCI63 G213xCI80 G251xCI80)

for genotype in "${GENOTYPES[@]}"; do
    echo "Submitting ${genotype}"
    sbatch job.run "${genotype}"
done
