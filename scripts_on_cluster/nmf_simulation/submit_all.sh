#!/bin/bash

# Usage: submit_all.sh <folder>
# <folder> can be a model name (expands to ./batch_scripts_<name>)
#          or a direct folder path (used as-is when it starts with . or /)
folder=${1:?Usage: submit_all.sh <folder_or_model_name>}

if [[ "$folder" == /* || "$folder" == ./* || "$folder" == ../* ]]; then
    scripts_dir="$folder"
else
    scripts_dir="./batch_scripts_$folder"
fi

files=($(ls $scripts_dir/*.run | sort))
file_count=${#files[@]}

echo -n "Are you sure you want to submit $file_count jobs from '$scripts_dir'? (y/Y to confirm) "
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
