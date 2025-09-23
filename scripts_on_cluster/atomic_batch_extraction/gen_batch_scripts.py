from pathlib import Path


project_dir = Path("~/biomechpose").expanduser()
template_path = project_dir / "scripts_on_cluster/atomic_batch_extraction/template.run"
script_output_dir = project_dir / "scripts_on_cluster/atomic_batch_extraction/batch_scripts"
log_dir = project_dir / "scripts_on_cluster/atomic_batch_extraction/logs"

output_basedir = project_dir / "bulk_data/pose_estimation/atomic_batches"

# Read template script
with open(template_path) as f:
    template_str = f.read()


def make_run_script(trial_dir):
    script_str = template_str \
        .replace("<<<JOB_NAME>>>", trial_dir.name) \
        .replace("<<<LOG_OUTPUT_FILE>>>", str(log_dir / trial_dir.name)) \
        .replace("<<<INPUT_BASEDIR>>>", str(trial_dir.absolute())) \
        .replace("<<<OUTPUT_DIR>>>", str(output_basedir / trial_dir.name))

    script_path = script_output_dir / f"{trial_dir.name}.run"
    with open(script_path, "w") as f:
        f.write(script_str)
    print(f"Script written to {script_path}")


if __name__ == "__main__":
    # Identify jobs to run
    trial_dirs = sorted(list((project_dir / "bulk_data/style_transfer/production/translated_videos/").glob("BO_Gal4_fly*_trial*")))

    for trial_dir in trial_dirs:
        make_run_script(trial_dir)

    print(f"{len(trial_dirs)} scripts written to {script_output_dir}")
