from pathlib import Path
from tqdm import tqdm

from biomechpose.simulate_nmf.data import load_kinematic_recording


# Job execution parameters
max_segs_per_run = 20

# Data filtering parameters
min_duration_frames = 10
filter_size = 5
filtered_frac_threshold = 0.5

# Define paths relevant to execution
project_dir = Path("~/biomechpose").expanduser()
template_path = project_dir / "scripts_on_cluster/nmf_simulation/template.run"
script_output_dir = project_dir / "scripts_on_cluster/nmf_simulation/batch_scripts"
log_dir = project_dir / "scripts_on_cluster/nmf_simulation/logs"
if list(script_output_dir.glob("*.run")):
    raise RuntimeError(
        "Batch scripts already found in the output directory. Empty them manually to "
        "be explicit about which machine-generated scipts to run."
    )
script_output_dir.mkdir(exist_ok=True, parents=True)
log_dir.mkdir(exist_ok=True, parents=True)

# Define paths relevant to data
output_basedir = project_dir / "bulk_data/nmf_rendering"
recorded_trials_dir = project_dir / "bulk_data/kinematic_prior/aymanns2022/trials/"
trial_data_files = sorted(list(recorded_trials_dir.glob("*.pkl")))

# Read template script
with open(template_path) as f:
    template_str = f.read()


def make_run_script(recorded_trial_path, segment_ids):
    segment_ids_str = [str(_id) for _id in segment_ids]
    job_name = f"{recorded_trial_path.stem}_segs{'-'.join(segment_ids_str)}"
    log_output_file = log_dir / f"{job_name}.out"
    trial_output_dir = output_basedir / recorded_trial_path.stem
    script_str = template_str \
        .replace("<<<JOB_NAME>>>", job_name) \
        .replace("<<<LOG_OUTPUT_FILE>>>", str(log_output_file)) \
        .replace("<<<RECORDED_TRIAL_PATH>>>", str(recorded_trial_path)) \
        .replace("<<<TRIAL_OUTPUT_DIR>>>", str(trial_output_dir)) \
        .replace("<<<SEGMENT_IDS>>>", str(" ".join(segment_ids_str)))

    script_path = script_output_dir / f"{job_name}.run"
    with open(script_path, "w") as f:
        f.write(script_str)
    print(f"Script written to {script_path}")


if __name__ == "__main__":
    # Identify jobs to run
    job_configs = []
    for in_path in tqdm(trial_data_files, desc="Building job specs"):
        # Check how many segments there are per trial
        kinematic_recording_segments = load_kinematic_recording(
            recording_path=in_path,
            min_duration_sec=0.2,
            input_timestep=0.01,
            filter_size=5,
            filtered_frac_threshold=0.5,
        )
        num_segments = len(kinematic_recording_segments)
        for start_idx in range(0, num_segments, max_segs_per_run):
            end_idx_exclusive = min(num_segments, start_idx + max_segs_per_run)
            segment_ids = list(range(start_idx, end_idx_exclusive))
            job_configs.append((in_path, segment_ids))

    # Generate scripts
    for recorded_trial_path, segment_ids in job_configs:
        make_run_script(recorded_trial_path, segment_ids)

    print(f"{len(job_configs)} scripts written to {script_output_dir}")
