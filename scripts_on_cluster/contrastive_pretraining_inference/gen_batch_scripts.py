import sys
from pathlib import Path


# Paths related to Slurm job configs
project_dir = Path("~/poseforge").expanduser()
template_path = project_dir / "scripts_on_cluster/contrastive_pretraining_inference/template.run"
script_output_dir = project_dir / "scripts_on_cluster/contrastive_pretraining_inference/batch_scripts"
log_dir = project_dir / "scripts_on_cluster/contrastive_pretraining_inference/logs"
if list(script_output_dir.glob("*.run")):
    raise RuntimeError(
        "Batch scripts already found in the output directory. Empty them manually to "
        "be explicit about which machine-generated scipts to run."
    )
script_output_dir.mkdir(exist_ok=True, parents=True)
log_dir.mkdir(exist_ok=True, parents=True)
curr_dir = Path(".").absolute()
if (
    curr_dir not in template_path.parents
    or curr_dir not in script_output_dir.parents
    or curr_dir not in log_dir.parents
):
    print(
        "One or more of project_dir, template_path, and log dir is not under the "
        "current directory. Double check if they're correct and rerun this script "
        "from a parent directory above them."
    )
    sys.exit(1)


# Read template script
with open(template_path) as f:
    template_str = f.read()


def make_run_script(fly_id):
    script_str = template_str \
        .replace("<<<FLY>>>", str(fly_id)) \
        .replace("<<<STDOUT_STDERR_FILE>>>", str(log_dir / f"fly{fly_id}.txt"))

    script_path = script_output_dir / f"fly{fly_id}.run"
    with open(script_path, "w") as f:
        f.write(script_str)
    print(f"Script written to {script_path}")


if __name__ == "__main__":
    # Make job scripts
    fly_ids = list(range(1, 6))
    for fly_id in fly_ids:
        make_run_script(fly_id)

    print(f"{len(fly_ids)} scripts written to {script_output_dir}")
