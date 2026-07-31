n_jobs=$(grep -c '' manifest.txt)
sbatch --array=1-"$n_jobs" "run_flygym_replay.run"
