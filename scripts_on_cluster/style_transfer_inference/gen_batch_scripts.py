import sys
from pathlib import Path
from biomechpose.style_transfer import parse_hyperparameters_from_trial_name


# Paths related to Slurm job configs
project_dir = Path("~/biomechpose").expanduser()
template_path = project_dir / "scripts_on_cluster/style_transfer_inference/template.run"
script_output_dir = project_dir / "scripts_on_cluster/style_transfer_inference/batch_scripts"
log_dir = project_dir / "scripts_on_cluster/style_transfer_inference/logs"
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


def make_run_script(model_name, epoch):
    hparams = parse_hyperparameters_from_trial_name(model_name)
    checkpoint_path = checkpoints_basedir / model_name / f"{epoch}_net_G.pth"
    job_name = f"inference_{model_name}_epoch{epoch}"
    output_video_filename = f"translated_{model_name}_epoch{epoch}.mp4"

    script_str = template_str \
        .replace("<<<NAME>>>", job_name) \
        .replace("<<<STDOUT_STDERR_FILE>>>", f"{log_dir}/{job_name}") \
        .replace("<<<CHECKPOINT_PATH>>>", str(checkpoint_path)) \
        .replace("<<<NGF>>>", str(hparams["ngf"])) \
        .replace("<<<NETG>>>", str(hparams["netG"])) \
        .replace("<<<TRAINING_BATCH_SIZE>>>", str(hparams["batsize"])) \
        .replace("<<<LAMBDA_GAN>>>", str(hparams["lambGAN"])) \
        .replace("<<<OUTPUT_VIDEO_FILENAME>>>", output_video_filename)

    script_path = script_output_dir / f"{job_name}.run"
    with open(script_path, "w") as f:
        f.write(script_str)
    print(f"Script written to {script_path}")


if __name__ == "__main__":
    # Define which models to use
    checkpoints_basedir = Path("bulk_data/style_transfer/production/trained_models/")
    models = [
        # (model_name, epoch)
        ("ngf16_netGsmallstylegan2_batsize2_lambGAN0.2", 121),
        ("ngf16_netGstylegan2_batsize4_lambGAN0.2", 200),
        ("ngf32_netGstylegan2_batsize2_lambGAN0.5-cont1", 161),
        ("ngf32_netGstylegan2_batsize4_lambGAN0.1", 161),
        ("ngf32_netGstylegan2_batsize4_lambGAN0.5", 141),
        ("ngf32_netGstylegan2_batsize4_lambGAN1.0", 161),
        ("ngf48_netGstylegan2_batsize4_lambGAN0.1", 141),
        ("ngf48_netGstylegan2_batsize2_lambGAN0.1", 141),
    ]

    # Make job scripts
    for model_name, epoch in models:
        make_run_script(model_name, epoch)

    print(f"{len(models)} scripts written to {script_output_dir}")
