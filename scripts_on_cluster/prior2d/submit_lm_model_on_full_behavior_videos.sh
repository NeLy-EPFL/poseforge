n_jobs=$(grep -c '' manifest.txt)
sbatch --array=1-"$n_jobs" "run_lm_model_on_full_behavior_videos.run"
