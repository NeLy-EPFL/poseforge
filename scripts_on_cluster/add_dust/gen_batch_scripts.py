"""Generate SLURM batch scripts that add dust to style-transferred videos.

Mirrors scripts_on_cluster/style_transfer_tiled_inference/gen_batch_scripts.py:
fill the template per job and write one ``.run`` file each, then submit with
``submit_all.sh``. One job per fly by default; adjust ``JOBS`` / ``FLY_IDS`` below.
"""

import sys
from pathlib import Path


# ---- Edit these paths for your cluster layout -------------------------------
work_user_dir = Path("/work/upramdya/stimpfli")
# Clean translated videos produced by the tiled style-transfer step.
input_basedir = work_user_dir / "poseforge/style_transfer/production/tiled_translated_videos"
# Where the dusty mirror tree is written.
output_basedir = work_user_dir / "poseforge/style_transfer/production/tiled_translated_videos_dusty"
# Artifact assets built by extract_dust and extract_walls.
artifacts_dir = work_user_dir / "poseforge/dust"
dust_library_path = artifacts_dir / "dust_library.npz"
dust_statistics_path = artifacts_dir / "dust_statistics.json"
wall_library_path = artifacts_dir / "wall_library.npz"
wall_statistics_path = artifacts_dir / "wall_statistics.json"

# Dust and wall presence/amount are drawn from their statistics files by default.
# SEED makes the run reproducible.
SEED = 0
VIDEO_GLOB = "translated_*.mp4"
# One job per fly keeps runtimes bounded; "*" would do everything in one job.
FLY_IDS = [5]
# -----------------------------------------------------------------------------


project_dir = Path("~/poseforge").expanduser()
template_path = project_dir / "scripts_on_cluster/add_dust/template.run"
script_output_dir = project_dir / "scripts_on_cluster/add_dust/batch_scripts"
log_dir = project_dir / "scripts_on_cluster/add_dust/logs"

if list(script_output_dir.glob("*.run")):
    raise RuntimeError(
        "Batch scripts already found in the output directory. Empty them manually "
        "to be explicit about which machine-generated scripts to run."
    )
script_output_dir.mkdir(exist_ok=True, parents=True)
log_dir.mkdir(exist_ok=True, parents=True)

current_dir = Path(".").absolute()
if (
    current_dir not in template_path.parents
    or current_dir not in script_output_dir.parents
    or current_dir not in log_dir.parents
):
    print(
        "One or more of project_dir, template_path, and log dir is not under the "
        "current directory. Double check they're correct and rerun from a parent "
        "directory above them."
    )
    sys.exit(1)

with open(template_path) as f:
    template_str = f.read()


def make_run_script(fly_id: int) -> None:
    """Generate one batch script for a single fly."""
    job_name = f"add_dust_fly{fly_id}"
    simulation_name_glob = f"BO_Gal4_fly{fly_id}_*"

    script_str = (
        template_str.replace("<<<NAME>>>", job_name)
        .replace("<<<STDOUT_STDERR_FILE>>>", f"{log_dir}/{job_name}")
        .replace("<<<INPUT_BASEDIR>>>", str(input_basedir))
        .replace("<<<OUTPUT_BASEDIR>>>", str(output_basedir))
        .replace("<<<DUST_LIBRARY_PATH>>>", str(dust_library_path))
        .replace("<<<DUST_STATISTICS_PATH>>>", str(dust_statistics_path))
        .replace("<<<WALL_LIBRARY_PATH>>>", str(wall_library_path))
        .replace("<<<WALL_STATISTICS_PATH>>>", str(wall_statistics_path))
        .replace("<<<VIDEO_GLOB>>>", VIDEO_GLOB)
        .replace("<<<SIMULATION_NAME_GLOB>>>", simulation_name_glob)
        .replace("<<<SEED>>>", str(SEED))
    )

    script_path = script_output_dir / f"{job_name}.run"
    with open(script_path, "w") as f:
        f.write(script_str)
    print(f"Script written to {script_path}")


if __name__ == "__main__":
    for fly_id in FLY_IDS:
        make_run_script(fly_id)
    print(f"{len(FLY_IDS)} scripts written to {script_output_dir}")
