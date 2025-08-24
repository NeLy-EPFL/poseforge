from pathlib import Path

"""
--ngf (# of gen filters in the last conv layer, default=64)   16 32
--netG (generator architecture, default=resnet_9blocks)       resnet_6blocks stylegan2 smallstylegan2
--batch_size (batch size, default=1)                          4 8
--lambda_GAN (weight for GAN loss, default=1.0)               0.5 1.0
"""

template_path = Path("/home/sibwang/biomechpose/scripts_on_cluster/20250822_parameter_sweep/template.run")
data_output_dir = Path("/scratch/sibwang/biomechpose/bulk_data/style_transfer/20250828_parameter_sweep/")
script_output_dir = Path("/home/sibwang/biomechpose/scripts_on_cluster/20250822_parameter_sweep/batch_scripts/")
checkpoint_basedir = data_output_dir / "checkpoints"
log_basedir = data_output_dir / "logs"
wandb_basedir = data_output_dir / "wandb"
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
    # Models with stylegan2
    netG = "stylegan2"
    for ngf in [32, 64]:
        for batch_size in [4, 8]:
            for lambda_GAN in [0.5, 1.0]:
                make_run_script(ngf, netG, batch_size, lambda_GAN)
                counter += 1

    # Model with smallstylegan2 for control
    netG = "smallstylegan2"
    batch_size = 8
    lambda_GAN = 0.5
    for ngf in [32, 64]:
        make_run_script(ngf, netG, batch_size, lambda_GAN)
        counter += 1

    print(f"{counter} scripts written to {script_output_dir}")
