"""Generate one Slurm batch script per trial for atomic-batch extraction.

Each generated script processes a single trial directory (one ``trial_name``
under ``all_trials_basedir``) and writes its atomic batches into its own
per-trial subdirectory under the size-suffixed output base, so that any
number of trial jobs can run in parallel without filename collisions.

Batch scripts and logs are routed to size-specific subfolders so that
multiple ``target_image_size`` configurations (e.g. 224 and 256) can
coexist without overwriting one another.
"""

import os
from pathlib import Path


# ============================================================
# Job-wide parameters (edit these to (re)generate scripts)
# ============================================================
# Set ``target_image_size`` to None to keep source resolution (no resize).
target_image_size: tuple[int, int] | None = (512, 512)
original_image_size: tuple[int, int] = (912, 912)
atomic_batch_nframes: int = 32
atomic_batch_nvariants_max: int = 5
minimum_time_diff_frames: int = 60
# Number of joblib workers inside each Slurm job. The template requests
# slightly more CPUs than this to leave headroom for ffmpeg / I/O.
n_jobs_per_trial: int = 16

# ============================================================
# Paths
# ============================================================
user = os.environ.get("USER", "user")
# $WORK on the cluster is /work/upramdya (shared, no user); we append USER ourselves
# below to mirror the original process_all.run convention.
work_dir = Path(os.environ.get("WORK", "/work/upramdya")) / user

all_trials_basedir = (
    work_dir
    / "poseforge/style_transfer/production/tiled_translated_videos/standard_gray"
    )
nmf_sim_rendering_basedir = work_dir / "poseforge/data/nmf_rendering"
output_basedir = work_dir / "poseforge/style_transfer/atomic_batches_nmf"

project_dir = Path("~/poseforge").expanduser()
template_path = project_dir / "scripts_on_cluster/atomic_batch_extraction/template.run"


# ============================================================
# Derive size suffix used to scope outputs, batch scripts, and logs
# ============================================================
if target_image_size is None:
    size_suffix = "nores"
    target_image_size_flag = ""
else:
    h, w = target_image_size
    size_suffix = f"{h}" if h == w else f"{h}x{w}"
    target_image_size_flag = f"--target-image-size {h} {w}"

script_output_dir = (
    project_dir
    / f"scripts_on_cluster/atomic_batch_extraction/batch_scripts/{size_suffix}"
)
log_dir = (
    project_dir / f"scripts_on_cluster/atomic_batch_extraction/logs/{size_suffix}"
)
output_dir = output_basedir / f"atomic_batches_{size_suffix}"

if list(script_output_dir.glob("*.run")):
    raise RuntimeError(
        f"Batch scripts already found in {script_output_dir}. Empty it manually "
        "to be explicit about which machine-generated scripts to run."
    )
script_output_dir.mkdir(exist_ok=True, parents=True)
log_dir.mkdir(exist_ok=True, parents=True)

# Read template script
with open(template_path) as f:
    template_str = f.read()


def make_run_script(trial_name: str) -> None:
    job_name = f"atomicbatch_{size_suffix}_{trial_name}"
    log_output_file = log_dir / f"{job_name}.out"
    trial_input_dir = all_trials_basedir / trial_name
    trial_output_dir = output_dir / trial_name
    orig_h, orig_w = original_image_size

    script_str = (
        template_str
        .replace("<<<JOB_NAME>>>", job_name)
        .replace("<<<LOG_OUTPUT_FILE>>>", str(log_output_file))
        .replace("<<<ATOMIC_BATCH_NFRAMES>>>", str(atomic_batch_nframes))
        .replace("<<<ATOMIC_BATCH_NVARIANTS_MAX>>>", str(atomic_batch_nvariants_max))
        .replace("<<<MINIMUM_TIME_DIFF_FRAMES>>>", str(minimum_time_diff_frames))
        .replace("<<<ORIGINAL_IMAGE_SIZE>>>", f"{orig_h} {orig_w}")
        .replace("<<<TARGET_IMAGE_SIZE_FLAG>>>", target_image_size_flag)
        .replace("<<<INPUT_BASEDIR>>>", str(trial_input_dir))
        .replace("<<<NMF_SIM_RENDERING_BASEDIR>>>", str(nmf_sim_rendering_basedir))
        .replace("<<<N_JOBS>>>", str(n_jobs_per_trial))
        .replace("<<<OUTPUT_DIR>>>", str(trial_output_dir))
    )

    script_path = script_output_dir / f"{job_name}.run"
    with open(script_path, "w") as f:
        f.write(script_str)
    print(f"Script written to {script_path}")


if __name__ == "__main__":
    assert all_trials_basedir.is_dir(), (
        f"`all_trials_basedir` {all_trials_basedir} is not a directory. "
        "Check that $WORK is mounted and the path is correct."
    )
    trial_names = sorted(
        [p.name for p in all_trials_basedir.iterdir() if p.is_dir()]
    )
    print(f"Found {len(trial_names)} trials in {all_trials_basedir}")

    for trial_name in trial_names:
        make_run_script(trial_name)

    print(f"\n{len(trial_names)} scripts written to {script_output_dir}")
    print(f"Logs will be written under                {log_dir}")
    print(f"Atomic-batch outputs will be written under {output_dir}")
