import torch
import logging
import sys
from tqdm import tqdm
from pathlib import Path

from poseforge.style_transfer import get_inference_pipeline, parse_hyperparameters_from_checkpoint_path
from poseforge.style_transfer.tiled_inference import process_simulation_tiled
from poseforge.util.sys import clear_memory_cache


def ensure_gpu_availability() -> None:
    if not torch.cuda.is_available():
        logging.warning(
            "CUDA device not available. Inference using CPU is extremely slow. "
            "Consider using a machine with a GPU."
        )
        sys.exit(1)


def find_all_simulation_paths(simulations_basedir: Path) -> list[Path]:
    """Find paths to all simulation directories under the base root directory.
    
    Looks for directories containing any of the expected video files.
    """
    all_simulation_paths = []
    for sim_dir in simulations_basedir.rglob("*"):
        if not sim_dir.is_dir():
            continue
        # Check if this directory contains video files
        if list(sim_dir.glob("*.mp4")):
            all_simulation_paths.append(sim_dir)
    return sorted(list(set(all_simulation_paths)))  # Remove duplicates and sort


def run_tiled_inference_all_simulations(
    checkpoint_path: str,
    simulations_basedir: str,
    output_basedir: str,
    input_video_filename: str = "processed_nmf_sim_render_flybody_grayscale.mp4",
    output_video_filename: str = "tiled_inference_output.mp4",
    patch_batch_size: int | None = None,
    weight_type: str = "cosine",
    randomize_seams: bool = True,
    seed: int | None = None,
    debug_mode: bool = False,
    device: str = "cuda",
    memory_cleanup_interval: int = 10,
    verbose: bool = False,
) -> None:
    """Run tiled style transfer inference on all NeuroMechFly simulations.

    Discovers all simulation directories under simulations_basedir and processes
    each one with tiled inference, using randomized seams and configurable
    weight blending by default.

    Args:
        checkpoint_path (str): Path to the trained model checkpoint file.
        simulations_basedir (str): Base directory containing simulation
            subdirectories. All directories nested under this base directory
            that contain video files will be processed.
        output_basedir (str): Base directory to save styled videos. The
            directory structure under this base directory will mirror that
            under `simulations_basedir`.
        input_video_filename (str): Filename of the input video within each
            simulation directory.
        output_video_filename (str): Filename of the styled output video
            within each output directory.
        patch_batch_size (int | None): Batch size for processing tiles.
            If None, the largest batch size that fits in GPU memory will be
            automatically detected.
        weight_type (str): Weight blending type: "uniform", "cosine", "gaussian",
            or "pyramid". Default is "cosine" for smooth transitions.
        randomize_seams (bool): Whether to randomize seam locations per frame.
            Default is True for maximum visual variety.
        seed (int | None): Random seed for reproducible randomization.
            If None, seams are non-deterministic.
        debug_mode (bool): Whether to draw tile outlines on output.
        device (str): Device to use for inference ("cuda" or "cpu").
        memory_cleanup_interval (int): Interval (in number of simulations
            processed) to perform memory cleanup.
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

    if device == "cuda":
        ensure_gpu_availability()

    # Index simulations to process
    all_simulation_paths = find_all_simulation_paths(simulations_basedir)
    print(f"Total number of simulations to process: {len(all_simulation_paths)}")
    if len(all_simulation_paths) == 0:
        print(f"No simulations found under {simulations_basedir}")
        return

    # Set up inference pipeline
    print(f"Getting inference pipeline for model at {checkpoint_path}...")
    model_hparams = parse_hyperparameters_from_checkpoint_path(checkpoint_path)
    model_hparams["preprocess_opt"]["preprocess"] = ""
    inference_pipeline = get_inference_pipeline(
        checkpoint_path, model_hparams, device=device
    )

    # Process each simulation
    print(f"Processing {len(all_simulation_paths)} simulations...")
    for i, simulation_path in enumerate(tqdm(all_simulation_paths, disable=None)):
        input_video_path = simulation_path / input_video_filename
        
        # Skip if input video doesn't exist
        if not input_video_path.is_file():
            logging.warning(f"Input video not found: {input_video_path}, skipping...")
            continue

        output_dir = Path(
            str(simulation_path).replace(str(simulations_basedir), str(output_basedir))
        )
        assert (
            output_basedir in output_dir.parents
        ), "Output directory is outside the specified output base directory"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_video_filename

        try:
            process_simulation_tiled(
                inference_pipeline=inference_pipeline,
                input_video_path=input_video_path,
                output_video_path=output_path,
                weight_type=weight_type,
                patch_batch_size=patch_batch_size,
                debug_mode=debug_mode,
                randomize_seams=randomize_seams,
                seed=seed,
                progress_bar=False,
                clear_memory_cache_after=False,
            )
        except Exception as e:
            logging.error(f"Failed to process {input_video_path}: {e}")
            continue

        # Periodic memory cleanup every once in a while
        if (i + 1) % memory_cleanup_interval == 0:
            logging.info(f"Processed {i + 1} simulations. Running memory cleanup...")
            clear_memory_cache(logging_level=logging.INFO)

    print(f"Finished processing {len(all_simulation_paths)} simulations.")


if __name__ == "__main__":
    import tyro

    tyro.cli(run_tiled_inference_all_simulations)
