from pathlib import Path
from tqdm import tqdm

from poseforge.neuromechfly.data import (
    load_kinematic_recording,
    load_deeperfly_recording,
    SOURCE_FORMAT_AYMANNS2022,
    SOURCE_FORMAT_DEEPERFLY,
)


# Which kinematic input format to generate jobs for:
#   SOURCE_FORMAT_AYMANNS2022 -> tethered Aymanns et al. 2022 recordings (*.pkl)
#   SOURCE_FORMAT_DEEPERFLY    -> freely-moving deeperfly recordings (*.h5)
source_format = SOURCE_FORMAT_AYMANNS2022

# For deeperfly only: which arena surface to keep ("floor", "lid", "wall" or
# "any"). Ignored for the Aymanns format. Use "floor" for floor-like walking.
deeperfly_surface = "floor"

# Job execution parameters
max_segs_per_run = 20

# Data filtering parameters
min_duration_sec = 0.2
filter_size = 5
filtered_frac_threshold = 0.5

# Define paths relevant to data
data_dir = Path("/work/upramdya/stimpfli/poseforge")
if source_format == SOURCE_FORMAT_DEEPERFLY:
    # deeperfly results live under one directory per recording, as
    # <recording>/deeperfly/results/<name>.h5. Confirm this glob matches the
    # layout on the cluster before submitting.
    deeperfly_recordings_dir = data_dir / "data/deeperfly/recordings"
    trial_data_files = sorted(deeperfly_recordings_dir.glob("*/deeperfly/results/*.h5"))
    # deeperfly joint angles are defined on the NeuroMechFly (nmf) model, so the
    # flybody model cannot be used with this format.
    use_flybody = False
else:
    recorded_trials_dir = data_dir / "data/aymanns2022/trials/"
    trial_data_files = sorted(recorded_trials_dir.glob("*.pkl"))
    use_flybody = "flybody" in str(recorded_trials_dir)

biomech_model = "flybody" if use_flybody else "nmf"
output_basedir = data_dir / f"data/{biomech_model}_rendering_{source_format}"

# Define paths relevant to execution
project_dir = Path("~/poseforge").expanduser()
template_path = project_dir / "scripts_on_cluster/nmf_simulation/template.run"
script_output_dir = project_dir / f"scripts_on_cluster/nmf_simulation/batch_scripts_{biomech_model}_{source_format}"
log_dir = project_dir / f"scripts_on_cluster/nmf_simulation/logs_{biomech_model}_{source_format}"
if list(script_output_dir.glob("*.run")):
    raise RuntimeError(
        "Batch scripts already found in the output directory. Empty them manually to "
        "be explicit about which machine-generated scipts to run."
    )
script_output_dir.mkdir(exist_ok=True, parents=True)
log_dir.mkdir(exist_ok=True, parents=True)

# Read template script
with open(template_path) as f:
    template_str = f.read()


def count_segments(recording_path):
    """Number of simulated segments in a recording, for the active source format."""
    if source_format == SOURCE_FORMAT_DEEPERFLY:
        segments = load_deeperfly_recording(
            recording_path=recording_path,
            min_duration_sec=min_duration_sec,
            surface=deeperfly_surface,
            filter_size=filter_size,
            filtered_frac_threshold=filtered_frac_threshold,
        )
    else:
        segments = load_kinematic_recording(
            recording_path=recording_path,
            min_duration_sec=min_duration_sec,
            input_timestep=0.01,
            filter_size=filter_size,
            filtered_frac_threshold=filtered_frac_threshold,
        )
    return len(segments)


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
        .replace("<<<USE_FLYBODY_FLAG>>>", str(use_flybody_flag)) \
        .replace("<<<SOURCE_FORMAT>>>", source_format) \
        .replace("<<<DEEPERFLY_SURFACE>>>", deeperfly_surface) \
        .replace("<<<BIOMECH_MODEL>>>", str(biomech_model))

    script_path = script_output_dir / f"{job_name}.run"
    with open(script_path, "w") as f:
        f.write(script_str)
    print(f"Script written to {script_path}")


if __name__ == "__main__":
    print(f"Source format: {source_format}")
    print(f"Found {len(trial_data_files)} recording file(s).")

    # Identify jobs to run
    job_configs = []
    num_segments_total = 0
    for in_path in tqdm(trial_data_files, desc="Building job specs"):
        # Check how many segments there are per trial
        num_segments = count_segments(in_path)
        if num_segments > 25:
            num_segments = 25
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
