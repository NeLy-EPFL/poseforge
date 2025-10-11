import h5py
from pathlib import Path
from tqdm import tqdm

from poseforge.pose_estimation import visualize_latent_trajectory


def visualize_simulation(
    inferred_latents_path: Path,
    output_dir: Path,
    training_stages: list[str],
    source_data_freq: int,
    play_speed: float = 0.1,
    trail_duration_sec: float = 0.5,
    output_fps: int = 30,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(inferred_latents_path, "r") as f:
        for feature in ["h_features_pooled", "z_features"]:
            for stage in training_stages:
                latent_space_data = f[f"{feature}/{stage}"][:]
                visualize_latent_trajectory(
                    latent_space_data=latent_space_data,
                    source_data_freq=source_data_freq,
                    play_speed=play_speed,
                    trail_duration_sec=trail_duration_sec,
                    output_fps=output_fps,
                    video_path=output_dir / f"{feature}_{stage}.mp4",
                    headless=True,
                )


if __name__ == "__main__":
    inference_output_basedir = Path(
        "bulk_data/pose_estimation/contrastive_pretraining/trial_20251011a/inference/"
    )
    output_basedir = inference_output_basedir / "latents_viz"
    example_sims = [
        "BO_Gal4_fly5_trial005/segment_003/subsegment_000",  # walking
        "BO_Gal4_fly5_trial005/segment_003/subsegment_001",  # grooming
        "BO_Gal4_fly5_trial005/segment_003/subsegment_002",  # both
    ]
    training_stages = ["untrained", "epoch000_step002000"]
    source_data_freq = 300  # Hz
    play_speed = 0.1  # fraction of real time
    trail_duration_sec = 0.5  # seconds
    output_fps = 30  # frames per second

    # Process each example simulation
    for sim_name in tqdm(example_sims):
        input_path = inference_output_basedir / sim_name / "contrastive_latents.h5"
        output_dir = output_basedir / sim_name
        visualize_simulation(
            inferred_latents_path=input_path,
            output_dir=output_dir,
            training_stages=training_stages,
            source_data_freq=source_data_freq,
            play_speed=play_speed,
            trail_duration_sec=trail_duration_sec,
            output_fps=output_fps,
        )
