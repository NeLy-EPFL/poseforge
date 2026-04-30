import json
import os
import sys
from pathlib import Path
from poseforge.style_transfer import parse_hyperparameters_from_checkpoint_path


scratch_user_dir = Path("/scratch") / os.environ.get("USER", "user")
data_base_dir = scratch_user_dir / "bulk_data"
data_base_dir_flybody = scratch_user_dir / "bulk_data_flybody"


def infer_data_config_from_checkpoint(checkpoint_path_str: str) -> tuple[str, str, str]:
    """Infer simulations_basedir, output_basedir, and input_video_filename from checkpoint.
    
    Reads train_options.json to determine data type (flybody, standard, etc.).
    Returns (simulations_basedir, output_basedir, input_video_filename).
    """
    checkpoint_path = Path(checkpoint_path_str)
    train_opt_path = checkpoint_path.parent.parent / "train_options.json"
    
    if not train_opt_path.is_file():
        raise FileNotFoundError(
            f"Could not find train_options.json at expected path: {train_opt_path}"
        )
    
    with open(train_opt_path, "r") as f:
        dataroot = json.load(f)["values"]["dataroot"]
    
    # Determine if flybody and render type
    is_flybody = "flybody" in dataroot
    
    if is_flybody:
        sim_base = str(data_base_dir_flybody)
        out_base = str(data_base_dir / "style_transfer/production/tiled_translated_videos/flybody")
        if "grayscale" in dataroot or "gray" in dataroot:
            video_file = "processed_nmf_sim_render_flybody_grayscale.mp4"
            out_base = str(data_base_dir / "style_transfer/production/tiled_translated_videos/flybody_gray")
        elif "base" in dataroot:
            video_file = "processed_nmf_sim_render_flybody_base.mp4"
            out_base = str(data_base_dir / "style_transfer/production/tiled_translated_videos/flybody_base")
        elif "link" in dataroot:
            video_file = "processed_nmf_sim_render_per_link_color.mp4"
            out_base = str(data_base_dir / "style_transfer/production/tiled_translated_videos/flybody_link")
        elif "segment_id" in dataroot or "segmentid" in dataroot:
            video_file = "processed_nmf_sim_segment_id.mp4"
            out_base = str(data_base_dir / "style_transfer/production/tiled_translated_videos/flybody_segment")
        elif "depth" in dataroot:
            video_file = "processed_nmf_sim_depth.mp4"
            out_base = str(data_base_dir / "style_transfer/production/tiled_translated_videos/flybody_depth")
        else:
            video_file = "processed_nmf_sim_render_flybody_grayscale.mp4"
    else:
        sim_base = str(data_base_dir)
        out_base = str(data_base_dir / "style_transfer/production/tiled_translated_videos/standard")
        if "gray" in dataroot or "grayscale" in dataroot:
            video_file = "processed_nmf_sim_render_grayscale.mp4"
            out_base = str(data_base_dir / "style_transfer/production/tiled_translated_videos/standard_gray")
        elif "base" in dataroot:
            video_file = "processed_nmf_sim_render_base.mp4"
            out_base = str(data_base_dir / "style_transfer/production/tiled_translated_videos/standard_base")
        elif "link" in dataroot:
            video_file = "processed_nmf_sim_render_per_link_color.mp4"
            out_base = str(data_base_dir / "style_transfer/production/tiled_translated_videos/standard_link")
        elif "segment_id" in dataroot or "segmentid" in dataroot:
            video_file = "processed_nmf_sim_segment_id.mp4"
            out_base = str(data_base_dir / "style_transfer/production/tiled_translated_videos/standard_segment")
        elif "depth" in dataroot:
            video_file = "processed_nmf_sim_depth.mp4"
            out_base = str(data_base_dir / "style_transfer/production/tiled_translated_videos/standard_depth")
        elif dataroot.endswith("aymanns2022_pseudocolor_spotlight_dataset"):
            video_file = "processed_nmf_sim_render_per_link_color.mp4"
            out_base = str(data_base_dir / "style_transfer/production/tiled_translated_videos/spotlight")
        else:
            video_file = "processed_nmf_sim_render_base.mp4"
    
    return sim_base, out_base, video_file


# Paths related to Slurm job configs
project_dir = Path("~/poseforge").expanduser()
template_path = project_dir / "scripts_on_cluster/style_transfer_tiled_inference/template.run"
script_output_dir = project_dir / "scripts_on_cluster/style_transfer_tiled_inference/batch_scripts"
log_dir = project_dir / "scripts_on_cluster/style_transfer_tiled_inference/logs"

if list(script_output_dir.glob("*.run")):
    raise RuntimeError(
        "Batch scripts already found in the output directory. Empty them manually to "
        "be explicit about which machine-generated scripts to run."
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


def make_run_script(model_name: str, epoch: int) -> None:
    """Generate a batch script for a model+epoch, auto-detecting data config."""
    checkpoint_path = checkpoints_basedir / model_name / model_name / f"{epoch}_net_G.pth"
    
    # Auto-detect data configuration from checkpoint
    sim_base, out_base, video_file = infer_data_config_from_checkpoint(str(checkpoint_path))
    
    job_name = f"tiled_inference_{model_name}_epoch{epoch}"

    script_str = template_str \
        .replace("<<<NAME>>>", job_name) \
        .replace("<<<STDOUT_STDERR_FILE>>>", f"{log_dir}/{job_name}") \
        .replace("<<<CHECKPOINT_PATH>>>", str(checkpoint_path)) \
        .replace("<<<SIMULATIONS_BASEDIR>>>", sim_base) \
        .replace("<<<OUTPUT_BASEDIR>>>", out_base) \
        .replace("<<<INPUT_VIDEO_FILENAME>>>", video_file)

    script_path = script_output_dir / f"{job_name}.run"
    with open(script_path, "w") as f:
        f.write(script_str)
    print(f"Script written to {script_path}")


if __name__ == "__main__":
    # Define which model checkpoints to process.
    # Data configuration (simulations_basedir, output_basedir, input_video_filename) is auto-detected from checkpoint.
    # Format: (checkpoint_path, model_name_for_job)

    checkpoints_basedir = Path("/mnt/upramdya_data/VAS/poseforge_checkpoints")
    
    
    jobs = [
        ("20260425_082514_lamG01_bs4_ngf48_flybody_gray", 334),
        ("20260425_082514_lamG01_bs4_ngf48_flybody_gray", 788),
        ("20260425_082346_lamG1_bs4_ngf32_flybody_gray", 609),
        ("20260425_082346_lamG1_bs4_ngf32_flybody_gray", 991),
        ("20260425_082309_lamG02_bs4_ngf16_flybody_gray", 961),
        ("20260425_082309_lamG02_bs4_ngf16_flybody_gray", 831),
        ("20260425_082136_lamG01_bs4_ngf322_flybody_gray", 665),
        ("20260425_082136_lamG01_bs4_ngf322_flybody_gray", 750)
    ]

    # # NMF
    # jobs = [
    #     ("20260425_082449_lamG01_bs4_ngf48_gray", 368),
    #     ("20260425_082410_lamG1_bs4_ngf32_gray", 706),
    #     ("20260425_082410_lamG1_bs4_ngf32_gray", 798),
    #     ("20260425_082410_lamG1_bs4_ngf32_gray", 986),
    #     ("20260425_082246_lamG02_bs4_ngf16_gray", 786),
    #     ("20260425_082246_lamG02_bs4_ngf16_gray", 816),
    #     ("20260425_082208_lamG01_bs4_ngf32_gray", 976),
    #     ("20260425_082208_lamG01_bs4_ngf32_gray", 740)
    # ]

    # Make job scripts
    for checkpoint_path, model_name in jobs:
        make_run_script(checkpoint_path, model_name)

    print(f"{len(jobs)} scripts written to {script_output_dir}")
