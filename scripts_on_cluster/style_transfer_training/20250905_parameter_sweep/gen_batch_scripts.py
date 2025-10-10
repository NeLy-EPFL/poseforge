from pathlib import Path
from os import getenv

username = getenv("USER")
assert username is not None and username != "", "USER environment variable not set"
campaign_dir = Path(f"/home/{username}/poseforge/scripts_on_cluster/style_transfer_training/20250905_parameter_sweep")
data_output_dir = Path(f"/scratch/{username}/poseforge/bulk_data/style_transfer/20250905_parameter_sweep/")
template_path = campaign_dir / "template.run"
script_output_dir = campaign_dir / "batch_scripts/"
checkpoint_basedir = data_output_dir / "checkpoints/"
log_basedir = data_output_dir / "logs/"
wandb_basedir = data_output_dir / "wandb/"
checkpoint_basedir.mkdir(parents=True, exist_ok=True)
log_basedir.mkdir(parents=True, exist_ok=True)
wandb_basedir.mkdir(parents=True, exist_ok=True)
script_output_dir.mkdir(parents=True, exist_ok=True)

if list(data_output_dir.glob("*.run")):
    raise RuntimeError(
        "Batch scripts already found in the output directory. Empty them manually to "
        "be explicit about which machine-generated scipts to run."
    )

with open(template_path) as f:
    template_str = f.read()


def make_run_script(ngf, netG, batch_size, lambda_GAN):
    run_name = (
        f"ngf{ngf}_netG{netG}_batsize{batch_size}_lambGAN{lambda_GAN}"
    )
    script_str = template_str \
        .replace("<<<NGF>>>", str(ngf)) \
        .replace("<<<NETG>>>", netG) \
        .replace("<<<BATCH_SIZE>>>", str(batch_size)) \
        .replace("<<<LAMBDA_GAN>>>", str(lambda_GAN)) \
        .replace("<<<CHECKPOINT_DIR>>>", str(checkpoint_basedir / run_name)) \
        .replace("<<<LOG_DIR>>>", str(log_basedir / run_name)) \
        .replace("<<<WANDB_DIR>>>", str(wandb_basedir / run_name)) \
        .replace("<<<OUTPUT_FILE>>>", str(script_output_dir.parent / f"outputs/{run_name}.out")) \
        .replace("<<<NAME>>>", run_name)

    filename = f"{run_name}.run"
    script_path = script_output_dir / filename
    with open(script_path, "w") as f:
        f.write(script_str)
    print(f"Script written to {script_path}")


if __name__ == "__main__":
    counter = 0
    
    # Using stylegan2 (6 blocks)
    netG = "stylegan2"
    for ngf in [16, 32, 48]:
        for batch_size in [2, 4]:
            for lambda_GAN in [0.2, 0.1]:
                make_run_script(ngf, netG, batch_size, lambda_GAN)
                counter += 1
    
    # Using smallstylegan2 (2 blocks)
    netG = "smallstylegan2"
    for ngf in [16, 32, 48]:
        for batch_size in [2]:
            for lambda_GAN in [0.2, 0.5, 1.0]:
                make_run_script(ngf, netG, batch_size, lambda_GAN)
                counter += 1

    print(f"{counter} scripts written to {script_output_dir}")
