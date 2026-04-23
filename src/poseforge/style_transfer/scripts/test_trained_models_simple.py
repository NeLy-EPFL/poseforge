from poseforge.style_transfer.scripts.test_trained_models import test_checkpoint
from pathlib import Path
import json
import torch

def easy_test_checkpoint(checkpoint_path: str, simulation_root: str):
    
    campaign_name = "test"
    checkpoint_path = Path(checkpoint_path)
    run_name = "test"
    trial_name = None #checkpoint_path.parent.name
    epoch = int(checkpoint_path.stem.split("_")[0])
    output_basedir = checkpoint_path.parent.parent.parent.parent
    # simulation_data_dirs = [Path(simulation_data_dirs).expanduser().resolve()]

    video_filename, is_flybody = parse_video_filename_from_checkpoint_path(checkpoint_path)
    
    # hardcode simulation data dirs for now
    simulation_root = Path(simulation_root)
    simulation_root = simulation_root / ("bulk_data_flybody" if is_flybody else "bulk_data")
    simulation_data_dirs = [
        simulation_root / "BO_Gal4_fly5_trial005/segment_001/subsegment_000",
        simulation_root / "BO_Gal4_fly3_trial005/segment_001/subsegment_000",
        simulation_root / "BO_Gal4_fly1_trial001/segment_001/subsegment_000",
    ]

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
                return "processed_nmf_sim_render_flybody_link.mp4", True
            elif "segment_id" in dataroot:
                return "processed_nmf_sim_segment_id.mp4", True
            elif "depth" in dataroot:
                return "processed_nmf_sim_render_depth.mp4", True
            else:
                raise ValueError(f"Could not parse video filename from dataroot: {dataroot}")
        else:
            if "gray" in dataroot or "grayscale" in dataroot:
                return "processed_nmf_sim_render_grayscale.mp4", False
            elif "base" in dataroot:
                return "processed_nmf_sim_render_base.mp4", False
            elif "link" in dataroot:
                return "processed_nmf_sim_render_link.mp4", False
            elif "segment_id" in dataroot:
                return "processed_nmf_sim_render_segment_id.mp4", False
            elif "depth" in dataroot:
                return "processed_nmf_sim_render_depth.mp4", False
            elif dataroot.endswith("aymanns2022_pseudocolor_spotlight_dataset"):
                return "processed_nmf_sim_render_link.mp4", False
            else:
                raise ValueError(f"Could not parse video filename from dataroot: {dataroot}")

    else:
        raise FileNotFoundError(
            f"Could not find train_options.json at expected path: {train_opt_path}"
        )

    

if __name__ == "__main__":
    import tyro
    tyro.cli(easy_test_checkpoint)