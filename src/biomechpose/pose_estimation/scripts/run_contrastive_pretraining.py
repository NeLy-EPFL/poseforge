import torch
from pathlib import Path
from torchvision.transforms import Compose, Grayscale

from biomechpose.pose_estimation.sampler import SyntheticFramesSampler
from biomechpose.pose_estimation.contrast_representation import (
    ContrastivePretrainingPipeline,
    RegNetFeatureExtractor,
    ContrastiveProjectionHead,
)
from biomechpose.util import set_random_seed, print_hardware_availability


if __name__ == "__main__":
    # Model hyperparameters
    # Number of different frames to include in each batch. Note that n_variants variants
    # of each frame will be included, so effective batch size = batch_size * n_vari
    batch_size = 64  # ~8.5GB GPU memory with n_variants=3
    # Whether to use pretrained weights for the feature extractor, or just use the
    # architecture with random initialization
    use_pretrained_backbone = True
    # Each two frames are at least 30 frames * (1 / 300 Hz) = 0.1s apart
    minimum_time_diff_frames = 30
    # Make frame grayscale, but preserve 3 channels to match pretrained RegNet input
    transform = Compose([Grayscale(num_output_channels=3)])
    # Layer sizes for the projection head
    projection_head_hidden_dim = 512
    projection_head_output_dim = 256
    # Training hyperparameters
    n_epochs = 10
    adam_kwargs = {"lr": 3e-4, "weight_decay": 1e-4}
    checkpoint_dir = Path("contrastive_pretraining/checkpoints")
    log_dir = Path("contrastive_pretraining/logs")
    checkpoint_interval_epochs = 1
    log_interval_steps = 1
    # Set random seed for reproducibility
    random_seed = 42
    # Limit size of dataset for rapid testing (None = use all available data)
    max_n_simulations: int | None = None  # 10
    max_n_models: int | None = 3

    # Set up IO, etc.
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)
    set_random_seed(random_seed)
    print_hardware_availability()

    # Define all training inputs
    input_basedir = Path("bulk_data/style_transfer/production/translated_videos")
    assert input_basedir.is_dir(), f"`input_basedir` {input_basedir} is not a directory"
    input_simulation_paths = sorted(list(input_basedir.rglob("subsegment_*")))
    if max_n_simulations is not None:
        input_simulation_paths = input_simulation_paths[:max_n_simulations]
    print(f"Found {len(input_simulation_paths)} simulations to use for pretraining")

    # Define image variants (i.e. style transfer output by different models)
    models = sorted(
        [path.stem for path in input_simulation_paths[0].glob("translated_*.mp4")]
    )
    if max_n_models is not None:
        models = models[:max_n_models]
    n_variants = len(models)
    print(f"Using {n_variants} input variants")

    # Index all video paths
    video_paths_by_sim_names = {}
    for sim_path in input_simulation_paths:
        sim_name = "/".join(sim_path.parts[-3:])
        video_paths_by_sim_names[sim_name] = [
            sim_path / f"{model}.mp4" for model in models
        ]

    # Create dataset
    dataset = SyntheticFramesSampler(
        video_paths_by_sim_names,
        batch_size=batch_size,
        sampling_stride=minimum_time_diff_frames,
        transform=transform,
        n_channels=3,
    )
    print(
        f"Dataset initialized with {len(dataset)} batches of samples from "
        f"{len(dataset.all_sim_names)} simulations. Total number of "
        f"usable frames: {dataset.total_n_frames} "
        f"({dataset.total_n_frames / 300:.2f} seconds at 300 FPS)."
    )

    # Create models
    feature_extractor = RegNetFeatureExtractor(pretrained=use_pretrained_backbone)
    print(f"Created feature extractor with output dim {feature_extractor.output_dim}")
    projection_head = ContrastiveProjectionHead(
        input_dim=feature_extractor.output_dim,
        hidden_dim=projection_head_hidden_dim,
        output_dim=projection_head_output_dim,
    )
    print(
        f"Created projection head with input dim {feature_extractor.output_dim}, "
        f"hidden dim {projection_head_hidden_dim}, "
        f"output dim {projection_head_output_dim}"
    )

    # Create training pipeline
    pipeline = ContrastivePretrainingPipeline(
        feature_extractor=feature_extractor,
        projection_head=projection_head,
        dataset=dataset,
        temperature=0.1,
        device="cuda",
        use_float16=True,
    )

    # Train the model
    pipeline.train(
        n_epochs=n_epochs,
        adam_kwargs=adam_kwargs,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        checkpoint_interval_epochs=checkpoint_interval_epochs,
        log_interval_steps=log_interval_steps,
        seed=random_seed,
    )
