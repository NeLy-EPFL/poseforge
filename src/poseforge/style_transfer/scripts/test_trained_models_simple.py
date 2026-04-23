from poseforge.style_transfer.scripts.test_trained_models import test_checkpoint
from pathlib import Path
import json
import torch

def easy_test_checkpoint(checkpoint_path: str, simulation_root: str, save_input_video: bool = False):
    
    campaign_name = "validation"
    checkpoint_path = Path(checkpoint_path)
    run_name = "validation"
    trial_name = None #checkpoint_path.parent.name
    epoch = int(checkpoint_path.stem.split("_")[0])
    output_basedir = checkpoint_path.parent.parent
    # simulation_data_dirs = [Path(simulation_data_dirs).expanduser().resolve()]

    video_filename, is_flybody = parse_video_filename_from_checkpoint_path(checkpoint_path)
    
    # hardcode simulation data dirs for now
    simulation_root = Path(simulation_root)
    simulation_root = simulation_root / ("bulk_data_flybody" if is_flybody else "bulk_data")
    simulation_data_dirs = [
        simulation_root / "BO_Gal4_fly5_trial005/segment_001/subsegment_000",
        simulation_root / "BO_Gal4_fly3_trial005/segment_001/subsegment_000",
    ]

    # if (output_basedir / run_name).exists():
    #     print(f"Output directory for run {run_name} already exists at {output_basedir / run_name}, skipping test.")
    #     return

    print("cuda" if torch.cuda.is_available() else "cpu")

    test_checkpoint(
        campaign_name=campaign_name,
        trial_name=trial_name,
        run_name=run_name,
        epoch=epoch,
        checkpoint_path=checkpoint_path,
        simulation_data_dirs=simulation_data_dirs,
        output_basedir=output_basedir,
        video_filename=video_filename,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_input_video = save_input_video,
        
    )

def parse_video_filename_from_checkpoint_path(checkpoint_path: Path) -> str:
    train_opt_path = checkpoint_path.parent.parent / "train_options.json"
    if train_opt_path.is_file():
        with open(train_opt_path, "r") as f:
            dataroot = json.load(f)["values"]["dataroot"]
        if "flybody" in dataroot:
            if "grayscale" in dataroot or "gray" in dataroot:
                return "processed_nmf_sim_render_flybody_grayscale.mp4", True
            elif "base" in dataroot:
                return "processed_nmf_sim_render_flybody_base.mp4", True
            elif "link" in dataroot:
                return "processed_nmf_sim_render_per_link_color.mp4", True
            elif "segment_id" in dataroot or "segmentid" in dataroot:
                return "processed_nmf_sim_segment_id.mp4", True
            elif "depth" in dataroot:
                return "processed_nmf_sim_depth.mp4", True
            else:
                raise ValueError(f"Could not parse video filename from dataroot: {dataroot}")
        else:
            if "gray" in dataroot or "grayscale" in dataroot:
                return "processed_nmf_sim_render_grayscale.mp4", False
            elif "base" in dataroot:
                return "processed_nmf_sim_render_base.mp4", False
            elif "link" in dataroot:
                return "processed_nmf_sim_render_per_link_color.mp4", False
            elif "segment_id" in dataroot or "segmentid" in dataroot:
                return "processed_nmf_sim_segment_id.mp4", False
            elif "depth" in dataroot:
                return "processed_nmf_sim_depth.mp4", False
            elif dataroot.endswith("aymanns2022_pseudocolor_spotlight_dataset"):
                return "processed_nmf_sim_render_per_link_color.mp4", False
            else:
                raise ValueError(f"Could not parse video filename from dataroot: {dataroot}")

    else:
        raise FileNotFoundError(
            f"Could not find train_options.json at expected path: {train_opt_path}"
        )

    
def test_all_checkpoints_in_directory(checkpoints_dir: str, simulation_root: str):
    checkpoints_dir = Path(checkpoints_dir)
    checkpoint_dirs = checkpoints_dir.glob("*")
    checkpoint_paths = []
    for ch_d in checkpoint_dirs:
        ch_d_paths = sorted(ch_d.rglob("[0-9]*G.pth"), key=lambda p: int(p.stem.split("_")[0]))
        if len(ch_d_paths) == 0:
            print(f"No checkpoint files found in directory {ch_d}, skipping.")
            continue
        selected_paths = ch_d_paths[-4:] if len(ch_d_paths) >= 4 else ch_d_paths
        for i in range(len(selected_paths)):
            if i == 0:
                checkpoint_paths.append((selected_paths[i], True))
            else:
                checkpoint_paths.append((selected_paths[i], False))
    # checkpoint_paths = sorted(checkpoints_dir.rglob("[0-9]*G.pth"), key=lambda p: int(p.stem.split("_")[0]))
    for checkpoint_path, save_input_video in checkpoint_paths:
        print(f"Testing checkpoint: {checkpoint_path}")
        easy_test_checkpoint(checkpoint_path, simulation_root, save_input_video=save_input_video)

if __name__ == "__main__":
    import tyro
    tyro.cli(test_all_checkpoints_in_directory)