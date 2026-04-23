import numpy as np
import torch
import json
import logging
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed

from poseforge.style_transfer import (
    get_inference_pipeline,
    process_simulation,
    parse_hyperparameters_from_trial_name,
    parse_hyperparameters_from_checkpoint_path,
)


def test_checkpoint(
    campaign_name: str,
    run_name: str,
    epoch: int,
    checkpoint_path: Path,
    simulation_data_dirs: list[Path],
    output_basedir: Path,
    trial_name: str | None = None,
    image_side_length: int = 256,
    batch_size: int | None = None,
    device: str | torch.device = "cuda",
    progress_bar: bool = False,
    video_filename: str = "",
    save_input_video: bool = False,
) -> None:
    """Visualize the performance of a model checkpoint by running inference
    on a set of NeuroMechFly-rendered behavior clips.

    Args:
        campaign_name: Name of the training campaign
            (e.g. "20250903_parameter_sweep")
        trial_name: Name of the training trial or None 
            (e.g. "ngf32_netGstylegan2_batsize4_lambGAN0.5")
        run_name: Name of the training run
            (e.g. "ngf32_netGstylegan2_batsize4_lambGAN0.5-cont1")
        epoch: Epoch number of the checkpoint (e.g. 100)
        checkpoint_path: Path to the checkpoint file
            (e.g. ".../ngf32_netGstylegan2_batsize4_lambGAN0.5-cont1/100_net_G.pth")
        simulation_data_dirs: List of paths to simulation directories,
            each containing a NeuroMechFly-rendered behavior clip
            (e.g. one named "processed_nmf_sim_render_colorcode_0.mp4")
        output_basedir: Base directory where output videos should be saved
            (file structure under this directory will be:
            {run_name}/epoch{epoch:03d}_examplesim{i:02d}.mp4)
        image_side_length: Side length (in px) to scale input images to.
        batch_size: Number of frames to process in a batch. If None, the
            maximum possible batch size will be automatically detected.
        device: Device to run inference on ("cpu" or "cuda")
        progress_bar: Whether to show a progress bar during inference
    """
    # Make output directory
    output_dir = output_basedir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse and save hyperparameters
    if trial_name is None:
        model_hparams = parse_hyperparameters_from_checkpoint_path(checkpoint_path)
    else:    
        model_hparams = parse_hyperparameters_from_trial_name(trial_name)
        model_hparams["image_side_length"] = image_side_length
    with open(output_dir / "metadata.json", "w") as f:
        metadata = {
            "hyperparameters": model_hparams,
            "campaign_name": campaign_name,
            "trial_name": trial_name,
            "run_name": run_name,
            "checkpoint_path": str(checkpoint_path),
            "simulation_data_dirs": [str(x.absolute()) for x in simulation_data_dirs],
        }
        json.dump(metadata, f, indent=2)

    # Get inference timeline
    inference_pipeline = get_inference_pipeline(checkpoint_path, model_hparams, device)

    # Run inference for each simulation run
    for i, sim_dir in enumerate(simulation_data_dirs):
        input_path = sim_dir / video_filename
        output_path = output_dir / f"epoch{epoch:03d}_examplesim{i:02d}.mp4"
        process_simulation(
            inference_pipeline, input_path, output_path, batch_size, progress_bar, save_input_video=save_input_video,
        )

def find_runs_within_trial_dir(trial_dir: Path) -> dict[str, Path]:
    """Given a trial directory, find all the runs that it might contain and
    return them as a dictionary (run name -> run path)."""
    # Find all train_opt.txt files, which are saved at the beginning of each run
    train_opt_paths = list(trial_dir.rglob("train_opt.txt"))
    run_paths = {}
    for train_opt_path in train_opt_paths:
        run_name = train_opt_path.parent.name
        run_paths[run_name] = train_opt_path.parent
    return run_paths


def find_available_checkpoints(
    run_dir: Path, network: str = "net_G"
) -> dict[int, Path]:
    """Given a run directory, find all saved network checkpoints."""
    checkpoint_paths_by_epoch = {}
    for path in run_dir.glob(f"*_{network}.pth"):
        epoch_str = path.stem.replace(f"_{network}", "")
        try:
            epoch = int(epoch_str)
            checkpoint_paths_by_epoch[epoch] = path
        except ValueError:
            continue

    return checkpoint_paths_by_epoch


def decide_checkpoints_to_test(
    checkpointed_epochs_lookup: dict[int, Path],
    epochs_interval: int,
) -> dict[int, Path]:
    """Given a dictionary of available checkpoints (epoch number -> path),
    and a desired epoch interval (i.e. run inference using checkpoints
    every N epochs), return a dictionary of selected checkpoints
    (epoch number -> path) so that the selected checkpoints are the
    closest available checkpoints to the desired epochs (note that a
    checkpoint might not have been saved at the exact desired epoch)."""
    available_epochs = np.array(sorted(checkpointed_epochs_lookup.keys()))
    start_epoch = available_epochs[0]
    end_epoch = available_epochs[-1] + 1
    desired_epochs = list(range(start_epoch, end_epoch + 1, epochs_interval))
    selected_checkpoints = {}
    for desired_epoch in desired_epochs:
        distance = np.abs(available_epochs - desired_epoch)
        closest_epoch = int(available_epochs[np.argmin(distance)])
        selected_checkpoints[closest_epoch] = checkpointed_epochs_lookup[closest_epoch]
    return selected_checkpoints


def define_all_checkpoints_to_test(
    training_campaign_dirs: list[Path],
    epochs_interval: int,
) -> list[tuple[str, str, str, int, Path]]:
    """Given a list of training campaign directories, scan through all
    trials and all runs within the trial and return a list of inference run
    specifications (each a tuple of
    (campaign_name, trial_name, run_name, epoch, checkpoint_path))."""
    all_checkpoints_to_test = []

    for campaign_dir in training_campaign_dirs:
        trial_dirs = [
            path for path in (campaign_dir / "checkpoints").iterdir() if path.is_dir()
        ]
        for trial_dir in trial_dirs:
            runs = find_runs_within_trial_dir(trial_dir)
            for run_name, run_dir in runs.items():
                # Decide which checkpoints to test
                available_checkpoints = find_available_checkpoints(run_dir)
                if len(available_checkpoints) == 0:
                    logging.warning(
                        f"No available checkpoints found in run directory: {run_dir}"
                    )
                    continue
                selected_checkpoints = decide_checkpoints_to_test(
                    available_checkpoints, epochs_interval=epochs_interval
                )

                # For each selected checkpoint, add specs to the list
                for epoch, checkpoint_path in selected_checkpoints.items():
                    specs = (
                        campaign_dir.name,
                        trial_dir.name,
                        run_name,
                        epoch,
                        checkpoint_path,
                    )
                    all_checkpoints_to_test.append(specs)

    return sorted(all_checkpoints_to_test)


if __name__ == "__main__":
    # ==================== Configuration ====================
    # Define directories to training campaigns
    training_data_basedir = Path(
        "~/Data/scitas_data/poseforge/bulk_data/style_transfer"
    ).expanduser()
    training_campaign_dirs = [
        training_data_basedir / "20250903_parameter_sweep",
        training_data_basedir / "20250905_parameter_sweep",
        training_data_basedir / "20250905_continued_training",
    ]

    # Define which behavior clip to use for testing
    simulation_dirs = [
        Path("bulk_data/nmf_rendering/BO_Gal4_fly1_trial001/segment_001/subsegment_000")
    ]

    # Define how often checkpoints should be sampled
    epochs_interval = 20

    # Define how inference runs should be executed.
    # Empirically, running on a single NVIDIA RTX 3080 Ti GPU is 5-10x
    # faster than a 16-core 11th Gen Intel Core i9-11900K CPU
    parallelism = "cuda"  # "cpu" or "cuda"

    # Define where the output should be saved
    output_basedir = Path("bulk_data/style_transfer/synthetic_output")

    # Set logging level
    logging.basicConfig(level=logging.WARNING)
    # ==================== End Configuration ====================

    # Index all inference runs required
    all_checkpoints_to_test = define_all_checkpoints_to_test(
        training_campaign_dirs, epochs_interval
    )
    num_campaigns = len(set([specs[0] for specs in all_checkpoints_to_test]))
    num_trials = len(set([specs[1] for specs in all_checkpoints_to_test]))
    num_runs = len(set([specs[2] for specs in all_checkpoints_to_test]))
    print(f"Total number of checkpoints to test: {len(all_checkpoints_to_test)}")

    # Check if specified parallelism option is available
    if parallelism == "cuda" and not torch.cuda.is_available():
        logging.error("CUDA device requested but not available. Using CPU instead.")
        parallelism = "cpu"
    if parallelism not in ["cpu", "cuda"]:
        logging.error(f"Unknown parallelism mode: {parallelism}. Using CPU instead.")
        parallelism = "cpu"

    # Run inference for each checkpoint
    if parallelism == "cpu":
        # Run many models in parallel on different CPU cores, but each
        # model will run without hardware acceleration
        def cpu_worker(specs):
            campaign_name, trial_name, run_name, epoch, checkpoint_path = specs
            test_checkpoint(
                campaign_name,
                trial_name,
                run_name,
                epoch,
                checkpoint_path,
                simulation_dirs,
                output_basedir,
                batch_size=4,
                device="cpu",
            )

        Parallel(n_jobs=-1)(
            delayed(cpu_worker)(specs) for specs in tqdm(all_checkpoints_to_test)
        )
    elif parallelism == "cuda":
        # Run models one by one, each on the GPU so that the computational
        # graph runs with hardware acceleration
        for specs in tqdm(all_checkpoints_to_test):
            campaign_name, trial_name, run_name, epoch, checkpoint_path = specs
            test_checkpoint(
                campaign_name,
                trial_name,
                run_name,
                epoch,
                checkpoint_path,
                simulation_dirs,
                output_basedir,
                batch_size=None,  # auto-detect
                device="cuda",
            )
