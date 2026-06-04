from pathlib import Path
from tqdm import tqdm

from poseforge.neuromechfly.data import load_kinematic_recording


# Job execution parameters
max_segs_per_run = 20

# Data filtering parameters
min_duration_frames = 10
filter_size = 5
filtered_frac_threshold = 0.5

# Define paths relevant to execution
project_dir = Path("~/poseforge").expanduser()
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
recorded_trials_dir = project_dir / "bulk_data/kinematic_prior/aymanns2022/trials/"
trial_data_files = sorted(list(recorded_trials_dir.glob("*.pkl")))

# Auto-detect whether to use the flybody model based on the input data path.
# We assume that any kinematic-prior directory whose path contains "flybody"
# (case-insensitive) was produced for the flybody model. The output base
# directory is selected accordingly so nmf and flybody renderings don't mix.
use_flybody = "flybody" in str(recorded_trials_dir).lower()
output_basedir = project_dir / (
    "bulk_data/nmf_rendering_flybody" if use_flybody else "bulk_data/nmf_rendering"
)
print(
    f"Auto-detected use_flybody={use_flybody} from {recorded_trials_dir}; "
    f"writing outputs to {output_basedir}"
)

# Read template script
with open(template_path) as f:
    template_str = f.read()


def make_run_script(recorded_trial_path, segment_ids, use_flybody):
    segment_ids_str = [str(_id) for _id in segment_ids]
    job_name = f"{recorded_trial_path.stem}_segs{'-'.join(segment_ids_str)}"
    log_output_file = log_dir / f"{job_name}.out"
    trial_output_dir = output_basedir / recorded_trial_path.stem
    use_flybody_flag = "--use-flybody" if use_flybody else "--no-use-flybody"
    script_str = template_str \
        .replace("<<<JOB_NAME>>>", job_name) \
        .replace("<<<LOG_OUTPUT_FILE>>>", str(log_output_file)) \
        .replace("<<<RECORDED_TRIAL_PATH>>>", str(recorded_trial_path)) \
        .replace("<<<TRIAL_OUTPUT_DIR>>>", str(trial_output_dir)) \
        .replace("<<<SEGMENT_IDS>>>", str(" ".join(segment_ids_str))) \
        .replace("<<<USE_FLYBODY_FLAG>>>", use_flybody_flag)

    script_path = script_output_dir / f"{job_name}.run"
    with open(script_path, "w") as f:
        f.write(script_str)
    print(f"Script written to {script_path}")


if __name__ == "__main__":
    # Identify jobs to run
    job_configs = []
    num_segments_total = 0
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
        num_segments_total += num_segments
        for start_idx in range(0, num_segments, max_segs_per_run):
            end_idx_exclusive = min(num_segments, start_idx + max_segs_per_run)
            segment_ids = list(range(start_idx, end_idx_exclusive))
            job_configs.append((in_path, segment_ids))
    print(f"Total number of recording sections to simulate: {num_segments_total}")

    # Generate scripts
    for recorded_trial_path, segment_ids in job_configs:
        make_run_script(recorded_trial_path, segment_ids, use_flybody)

    print(f"{len(job_configs)} scripts written to {script_output_dir}")
