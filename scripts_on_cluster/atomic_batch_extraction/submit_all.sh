#!/bin/bash
#
# Submit all generated trial-level batch scripts for a given target image size.
#
# Usage:
#     bash submit_all.sh <size_suffix>
# e.g.
#     bash submit_all.sh 256
#     bash submit_all.sh nores
#
# The <size_suffix> must match a subdirectory of ./batch_scripts/ created by
# gen_batch_scripts.py (e.g. "256", "224", "256x192", "nores", ...).

if [ -z "$1" ]; then
    echo "Usage: $0 <size_suffix>"
    echo "Available subfolders under ./batch_scripts/:"
    ls -1 ./batch_scripts 2>/dev/null || echo "  (none)"
    exit 1
fi

size_suffix="$1"
scripts_dir="./batch_scripts/${size_suffix}"

if [ ! -d "$scripts_dir" ]; then
    echo "Error: $scripts_dir does not exist."
    echo "Run gen_batch_scripts.py with target_image_size matching '${size_suffix}' first."
    exit 1
fi

files=($(ls $scripts_dir/*.run 2>/dev/null | sort))
file_count=${#files[@]}

if [ "$file_count" -eq 0 ]; then
    echo "No .run scripts found in $scripts_dir."
    exit 1
fi

echo -n "Are you sure you want to submit $file_count jobs from ${scripts_dir}? (y/Y to confirm) "
read -r confirmation
if [[ "$confirmation" != "y" && "$confirmation" != "Y" ]]; then
    echo "Submission canceled."
    exit 1
fi

for file in "${files[@]}"; do
    echo "Submitting $file"
    sbatch $file
done

echo "Submitted $file_count files to the scheduler"
