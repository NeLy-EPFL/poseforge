import torch
import logging
import sys
from tqdm import tqdm
from pathlib import Path
from pvio.io import read_frames_from_video

from poseforge.style_transfer import (
    get_inference_pipeline,
    parse_hyperparameters_from_checkpoint_path,
    process_simulation,
)
from poseforge.util.sys import clear_memory_cache


def resolve_trained_resolution(
    checkpoint_path: Path, requested_image_side_length: int
) -> tuple[int, dict | None]:
    """Determine the inference resolution and preprocessing for a checkpoint.

    The non-tiled inference path historically hard-coded the input resolution to
    ``requested_image_side_length`` (default 256) and a fixed
    ``resize_and_crop`` preprocessing, ignoring the resolution the CUT model was
    actually trained at. A model trained at a higher resolution would therefore
    still be run at 256.

    This helper mirrors the tiled inference script: it reads ``crop_size`` /
    ``load_size`` / ``preprocess`` from the trained model's ``train_options.json``
    (via :func:`parse_hyperparameters_from_checkpoint_path`) and uses the trained
    ``crop_size`` as the input side length. If ``train_options.json`` is absent it
    falls back to ``requested_image_side_length`` and the legacy preprocessing.

    A warning is emitted if the user-supplied ``requested_image_side_length``
    conflicts with the resolution the model was trained at.

    Returns:
        A tuple ``(image_side_length, preprocess_opt)`` where ``preprocess_opt``
        is either a dict suitable for ``CUTPreprocessOptions(**preprocess_opt)``
        or ``None`` to use the legacy default preprocessing.
    """
    try:
        trained_hparams = parse_hyperparameters_from_checkpoint_path(checkpoint_path)
    except FileNotFoundError:
        logging.warning(
            "Could not find train_options.json next to checkpoint %s; falling "
            "back to the requested input resolution of %d px (the model's "
            "trained crop_size is unknown).",
            checkpoint_path,
            requested_image_side_length,
        )
        return requested_image_side_length, None

    trained_side_length = trained_hparams["image_side_length"]
    preprocess_opt = trained_hparams.get("preprocess_opt")

    if trained_side_length != requested_image_side_length:
        logging.warning(
            "Requested image_side_length=%d but the model was trained at "
            "crop_size=%d. Using the trained resolution %d px so the generator "
            "runs at the resolution it was trained for. Pass "
            "image_side_length=%d to silence this warning.",
            requested_image_side_length,
            trained_side_length,
            trained_side_length,
            trained_side_length,
        )

    return trained_side_length, preprocess_opt


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
        image_side_length (int): Requested side length (in pixels) of input
            images. This is only used as a fallback: if the model's
            train_options.json is found next to the checkpoint, the resolution
            (crop_size) and preprocessing the model was actually trained at are
            used instead, and a warning is logged if this value disagrees with
            the trained crop_size.
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

    # Set up inference pipeline. Honor the resolution / preprocessing the CUT
    # model was actually trained at (parsed from its train_options.json) instead
    # of hard-coding 256, so a model trained at higher resolution runs at that
    # resolution. Falls back to the requested image_side_length if unavailable.
    print(f"Getting inference pipeline for model at {checkpoint_path}...")
    resolved_image_side_length, preprocess_opt = resolve_trained_resolution(
        checkpoint_path, image_side_length
    )
    logging.info(
        "Running style-transfer inference at %d px (preprocess: %s).",
        resolved_image_side_length,
        (preprocess_opt or {}).get("preprocess", "resize_and_crop (default)"),
    )
    print(
        f"Style-transfer inference resolution: {resolved_image_side_length} px "
        f"(input images are resized/cropped to this before the generator)."
    )
    model_hparams = {
        "ngf": ngf,
        "netG": netG,
        "batsize": training_batch_size,
        "lambGAN": lambGAN,
        "image_side_length": resolved_image_side_length,
        "preprocess_opt": preprocess_opt,
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
