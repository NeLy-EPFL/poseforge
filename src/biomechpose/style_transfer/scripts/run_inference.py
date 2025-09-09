import torch
import logging
import sys
from tqdm import tqdm
from pathlib import Path

from biomechpose.style_transfer import (
    get_inference_pipeline,
    process_simulation,
    parse_hyperparameters_from_trial_name,
)
from biomechpose.util import read_frames_from_video, clear_memory_cache


def ensure_gpu_availability():
    if not torch.cuda.is_available():
        logging.warning(
            "CUDA device not available. Inference using CPU is extremely slow. "
            "Consider using a machine with a GPU."
        )
        sys.exit(1)


def get_all_simulation_paths(nmf_renderings_basedir):
    """Given the base directory containing all NMF simulations, return a
    sorted list of all simulation (i.e. subsegment) directories."""
    all_simulation_paths = []
    for motion_prior_trial_dir in nmf_renderings_basedir.iterdir():
        if not motion_prior_trial_dir.is_dir():
            logging.warning(
                f"Unexpected file under nmf_renderings_basedir: "
                f"{motion_prior_trial_dir}. Skipping."
            )
            continue
        for segment_dir in motion_prior_trial_dir.glob("segment_*"):
            if not segment_dir.is_dir():
                logging.warning(
                    f"Unexpected file under motion prior directory: {segment_dir}. "
                    f"Skipping."
                )
                continue
            for subsegment_dir in segment_dir.glob("subsegment_*"):
                if not subsegment_dir.is_dir():
                    logging.warning(
                        f"Unexpected file under simulated segment directory: "
                        f"{subsegment_dir}. Skipping."
                    )
                    continue
                all_simulation_paths.append(subsegment_dir)
    all_simulation_paths = sorted(all_simulation_paths)
    return all_simulation_paths


def get_inference_pipeline_and_max_batch_size(
    checkpoint_path, example_simulation_path, device
):
    """Set up the inference pipeline and auto-detect the maximum batch size
    that fits in GPU memory."""
    # Parse model hyperparameters from the checkpoint path
    run_name = checkpoint_path.parent.name
    model_hparams = parse_hyperparameters_from_trial_name(run_name)

    # Get the inference pipeline
    inference_pipeline = get_inference_pipeline(checkpoint_path, model_hparams, device)
    logging.info(f"Loaded model from {checkpoint_path} with hyperparameters:")
    for k, v in model_hparams.items():
        logging.info(f"  {k}: {v}")

    # Auto-detect largest batch size that fits in GPU memory
    example_input_video_path = (
        example_simulation_path / "processed_nmf_sim_render_colorcode_0.mp4"
    )
    video_frames, fps = read_frames_from_video(example_input_video_path)
    max_batch_size = inference_pipeline.detect_max_batch_size(
        input_image_shape=video_frames[0].shape, exponential=True
    )
    logging.info(f"Detected maximum batch size: {max_batch_size}")

    return inference_pipeline, max_batch_size


if __name__ == "__main__":
    # Define model and data paths
    models = [
        # (model_name, epoch)
        ("ngf16_netGsmallstylegan2_batsize2_lambGAN0.2", 121),
        ("ngf16_netGstylegan2_batsize4_lambGAN0.2", 200),
        ("ngf32_netGstylegan2_batsize2_lambGAN0.5-cont1", 161),
        ("ngf32_netGstylegan2_batsize4_lambGAN0.1", 161),
        ("ngf32_netGstylegan2_batsize4_lambGAN0.5", 141),
        ("ngf32_netGstylegan2_batsize4_lambGAN1.0", 161),
        ("ngf48_netGstylegan2_batsize4_lambGAN0.1", 141),
    ]
    checkpoints_basedir = Path("bulk_data/style_transfer/production/trained_models/")
    nmf_renderings_basedir = Path("bulk_data/nmf_rendering/")
    output_basedir = Path("bulk_data/style_transfer/production/translated_sim_videos/")

    # Set logging level
    logging.basicConfig(level=logging.INFO)

    # Check if GPU is available
    ensure_gpu_availability()
    device = "cuda"

    # Index all simulations to process
    all_simulation_paths = get_all_simulation_paths(nmf_renderings_basedir)
    print(f"Total number of simulations to process: {len(all_simulation_paths)}")

    # Run inference using each model
    print(f"Total number of models to use: {len(models)}")
    for model_name, epoch in models:
        print(f"Running inference using model {model_name}, epoch {epoch}")
        checkpoint_path = checkpoints_basedir / model_name / f"{epoch}_net_G.pth"

        # Parse model hyperparameters from the checkpoint path
        model_training_run_name = checkpoint_path.parent.name
        model_hparams = parse_hyperparameters_from_trial_name(model_training_run_name)

        # Set up inference pipeline
        inference_pipeline, max_batch_size = get_inference_pipeline_and_max_batch_size(
            checkpoint_path, all_simulation_paths[0], device
        )

        # Process each simulation
        for i, simulation_path in enumerate(tqdm(all_simulation_paths)):
            trial, segment, subsegment = simulation_path.parts[-3:]
            input_video_path = (
                simulation_path / "processed_nmf_sim_render_colorcode_0.mp4"
            )
            output_dir = output_basedir / trial / segment / subsegment
            output_dir.mkdir(parents=True, exist_ok=True)
            output_video_path = (
                output_dir / f"styled_video_{model_training_run_name}_epoch{epoch}.mp4"
            )

            process_simulation(
                inference_pipeline,
                input_video_path,
                output_video_path,
                batch_size=max_batch_size,
                progress_bar=False,
            )

            # Periodic memory cleanup every 10 simulations
            if (i + 1) % 10 == 0:
                logging.info(
                    f"Processed {i + 1} simulations. Running memory cleanup..."
                )
                clear_memory_cache(logging_level=logging.INFO)
