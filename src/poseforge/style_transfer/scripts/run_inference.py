import torch
import logging
import sys
from tqdm import tqdm
from pathlib import Path
from pvio.video_io import read_frames_from_video

from poseforge.style_transfer import get_inference_pipeline, process_simulation
from poseforge.util.sys import clear_memory_cache


def ensure_gpu_availability() -> None:
    if not torch.cuda.is_available():
        logging.warning(
            "CUDA device not available. Inference using CPU is extremely slow. "
            "Consider using a machine with a GPU."
        )
        sys.exit(1)


def find_all_simulation_paths(nmf_renderings_basedir: Path) -> list[Path]:
    """Find paths to all NeuroMechFly simulations directories under the
    base root directory."""
    all_simulation_paths = [
        file.parent
        for file in nmf_renderings_basedir.rglob("processed_simulation_data.h5")
    ]
    return sorted(list(all_simulation_paths))


def run_inference_cli(
    checkpoint_path: str,
    simulations_basedir: str,
    output_basedir: str,
    ngf: int,
    netG: str,
    training_batch_size: int,
    lambGAN: float,
    image_side_length: int = 256,
    input_video_filename: str = "processed_nmf_sim_render_colorcode_0.mp4",
    output_video_filename: str = "domain_translated_video.mp4",
    inference_batch_size: int | None = None,
    device: str = "cuda",
    memory_cleanup_interval: int = 10,
    verbose: bool = False,
) -> None:
    """Run style transfer inference on NeuroMechFly simulations to make the
    renderings look like Spotelight behavior recordings.

    Args:
        checkpoint_path (str): Path to the trained model checkpoint file.
        simulations_basedir (str): Base directory containing NeuroMechFly
            simulation subdirectories. All directories nested under this base
            base directory that contain a "processed_kinematic_states.pkl"
            file will be processed.
        output_basedir (str): Base directory to save styled videos. The
            directory structure under this base directory will mirror that
            under `simulations_basedir`.
        ngf (int): Number of generator filters in the last conv layer. This
            must match the value used during training.
        netG (str): Type of generator architecture. This must match the
            architecture used during training.
        training_batch_size (int): Batch size used during training.
        lambGAN (float): Weight for the GAN loss during training.
        image_side_length (int): Side length (in pixels) of input images.
        input_video_filename (str): Filename of the input video within each
            simulation directory. For example,
            "processed_nmf_sim_render_colorcode_0.mp4", which is the
            pseudocolor rendering of the NeuroMechFly simulations with leg
            segments shown in artificially bright colors to enhance
            contrast against the body. This must be the same rendering
            resolution used during training.
        output_video_filename (str): Filename of the styled output video
            within each output directory.
        inference_batch_size (int | None): Batch size to use during
            inference. This is different from the training batch size; this
            number only affects inference speed and memory usage. If None,
            the largest batch size that fits in GPU memory will be
            automatically detected.
        device (str): Device to use for inference ("cuda" for GPU vs.
            "cpu" for CPU). It is HIGHLY recommended to use a GPU, as
            inference on CPU is EXTREMELY slow.
        memory_cleanup_interval (int): Interval (in number of simulations
            processed) to perform memory cleanup. This can help avoid
            out-of-memory errors when processing a large number of
            simulations.
        verbose (bool): Whether to print detailed logs.
    """
    checkpoint_path = Path(checkpoint_path)
    simulations_basedir = Path(simulations_basedir)
    output_basedir = Path(output_basedir)

    # Set logging level
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Index simulations to process
    all_simulation_paths = find_all_simulation_paths(Path(simulations_basedir))
    print(f"Total number of simulations to process: {len(all_simulation_paths)}")
    if len(all_simulation_paths) == 0:
        return

    # Set up inference pipeline
    print(f"Getting inference pipeline for model at {checkpoint_path}...")
    model_hparams = {
        "ngf": ngf,
        "netG": netG,
        "batsize": training_batch_size,
        "lambGAN": lambGAN,
        "image_side_length": image_side_length,
    }
    inference_pipeline = get_inference_pipeline(checkpoint_path, model_hparams, device)
    if inference_batch_size is None:
        # Auto-detect largest batch size that fits in GPU memory
        example_input_video_path = all_simulation_paths[0] / input_video_filename
        video_frames, fps = read_frames_from_video(
            example_input_video_path, frame_indices=[0]
        )
        inference_batch_size = inference_pipeline.detect_max_batch_size(
            input_image_shape=video_frames[0].shape, exponential=True, end=512
        )

    # Process each simulation
    print(f"Processing {len(all_simulation_paths)} simulations...")
    for i, simulation_path in enumerate(tqdm(all_simulation_paths, disable=None)):
        input_video_path = simulation_path / "processed_nmf_sim_render_colorcode_0.mp4"
        output_dir = Path(
            str(simulation_path).replace(str(simulations_basedir), str(output_basedir))
        )
        assert (
            output_basedir in output_dir.parents
        ), "Output directory is outside the specified output base directory"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_video_filename

        process_simulation(
            inference_pipeline,
            input_video_path,
            output_path,
            batch_size=inference_batch_size,
            progress_bar=False,
        )

        # Periodic memory cleanup every once in a while
        if (i + 1) % memory_cleanup_interval == 0:
            logging.info(f"Processed {i + 1} simulations. Running memory cleanup...")
            clear_memory_cache(logging_level=logging.INFO)


if __name__ == "__main__":
    import tyro

    tyro.cli(run_inference_cli)

    # * Example call
    # model_name = "ngf16_netGsmallstylegan2_batsize2_lambGAN0.2"
    # epoch = 121
    # run_inference_cli(
    #     checkpoint_path=f"bulk_data/style_transfer/production/trained_models/{model_name}/{epoch}_net_G.pth",
    #     simulations_basedir="bulk_data/nmf_rendering/BO_Gal4_fly1_trial001/",
    #     output_basedir="bulk_data/style_transfer/production/translated_videos/BO_Gal4_fly1_trial001",
    #     ngf=16,
    #     netG="smallstylegan2",
    #     training_batch_size=2,
    #     lambGAN=0.2,
    #     input_video_filename="processed_nmf_sim_render_colorcode_0.mp4",
    #     output_video_filename=f"translated_{model_name}_epoch{epoch}.mp4",
    #     inference_batch_size=None,  # auto-detect
    #     device="cuda",
    #     memory_cleanup_interval=10,
    #     verbose=True,
    # )
