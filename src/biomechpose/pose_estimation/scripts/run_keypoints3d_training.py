import logging

logging_level = logging.INFO
logging.basicConfig(
    level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s"
)

import torch
from dataclasses import dataclass
from pathlib import Path

from biomechpose.pose_estimation.data import init_atomic_dataset_and_dataloader
from biomechpose.pose_estimation.feature_extractor import ResNetFeatureExtractor
from biomechpose.pose_estimation.keypoints_3d import (
    Pose2p5DModel,
    Pose2p5DLoss,
    Pose2p5DPipeline,
)
from biomechpose.util import set_random_seed, get_hardware_availability


@dataclass
class SamplingConfig:
    # Numbers of samples (frames) in each pre-extracted atomic batch
    atomic_batch_nsamples: int
    # Number of variants (synthetic images made by different style transfer models)
    atomic_batch_nvariants: int
    # Number of different frames to include in each batch. Note that n_variants variants
    # of each frame will be included, so effective batch size =
    # train_batch_size * n_variants.
    # This must be a multiple of `atomic_epoch_nsamples` in `AtomicBatchDataset`.
    train_batch_size: int
    # Validation batch size. Can be much smaller than train_batch_size. Must be
    # a multiple of `atomic_epoch_nsamples` in `AtomicBatchDataset`
    val_batch_size: int = 128
    # Number of workers for data loading. Use number of CPU cores if None.
    num_workers: int | None = None


@dataclass
class ModelConfig:
    # Feature extractor weights. Can be a path to the (contrastively) pretrained weights
    # or "IMAGENET1K_V1"
    feature_extractor_weights: str
    # Model weights, optional. If provided, the model will be initialized from these
    # weights (in which case feature_extractor_weights is ignored).
    model_weights: str | None = None
    # Number of hidden dimensions in the contrastive projection head (3-layer MLP)
    projection_head_hidden_dim: int = 512
    # Number of output dimensions in the contrastive projection head (3-layer MLP)
    projection_head_output_dim: int = 256
    # Number of body keypoints to detect
    n_keypoints: int = 32
    # Number of bins to quantize depth values (distances from camera) into
    depth_n_bins: int = 64
    # Minimum depth (distance from camera) in mm
    depth_min: float = 0.0
    # Maximum depth (distance from camera) in mm
    depth_max: float = 2.0
    # Temperature param to regulate the "softness" of the predicted x-y heatmaps
    xy_temperature: float = 0.8
    # Temperature param to regulate the "softness" of the predicted depth distributions
    depth_temperature: float = 0.8
    # Number of layers in the core upsampling pathway of the 3D pose estimation model
    upsample_n_layers: int = 3
    # Number of hidden channels in each deconv layer in the core upsampling pathway
    upsample_n_hidden_channels: int = 256
    # Number of hidden channels in the head that predicts depth distributions
    depth_n_hidden_channels: int = 256
    # Method to compute confidence scores from predicted distr ("entropy" for entropy
    # over predicted distr, "peak" for highest predicted probability in the distr)
    confidence_method: str = "entropy"
    # Number of groups in GroupNorm layers
    groupnorm_n_groups: int = 32
    # Std dev for initializing of final layers in x-y and depth heads (no ReLU after)
    pose_head_init_std: float = 1e-3


@dataclass
class LossConfig:
    # Loss function for x-y heatmaps ("mse" for simple mean squared error between
    # predicted and target heatmaps, "kl" for KL divergence between predicted and
    # target heatmaps treated as probability distributions)
    heatmap_loss_func: str = "kl"
    # Std dev (in heatmap pixels) of Gaussian used to generate target x-y heatmaps. Note
    # that a pixel on the heatmap may correspond to multiple pixels in the input image.
    heatmap_sigma: float = 2.0
    # Std dev (in depth bins) of Gaussian used to generate soft target depth distr
    depth_sigma_bins: float = 1.0
    # Weight for x-y heatmap loss term
    xy_loss_weight: float = 4.0
    # Weight for depth cross-entropy loss term
    depth_ce_loss_weight: float = 1.0
    # Weight for depth L1 loss term
    depth_l1_loss_weight: float = 0.25
    # Whether to clamp predicted depth values to be within [depth_min, depth_max]
    # (a warning will always be issued if predicted depths are out of bounds)
    clamp_labels: bool = True


@dataclass
class DataConfig:
    # Paths to training data (recursively containing atomic batches)
    train_data_dirs: list[str]
    # Paths to validation data (recursively containing atomic batches)
    val_data_dirs: list[str]
    # Frame size (height, width)
    image_size: tuple[int, int] = (256, 256)
    # Base depth of all keypoints in mm. The model will predict the relative depth, i.e.
    # depth_label - base_depth, instead of the absolute depth.
    depth_offset: float = 100.0


@dataclass
class TrainingConfig:
    # Total number of epochs to train for
    num_epochs: int
    # Random seed for reproducibility
    seed: int = 42
    # Learning rate for Adam optimizer
    adam_lr: float = 3e-4
    # Weight decay for Adam optimizer
    adam_weight_decay: float = 1e-4


@dataclass
class OutputConfig:
    # Base directory to save logs and model checkpoints
    output_basedir: str
    # Log training metrics every N steps
    logging_interval: int = 10
    # Save model checkpoint every N steps (NOT EPOCHS!)
    checkpoint_interval: int = 500
    # Run validation every N steps (NOT EPOCHS!)
    validation_interval: int = 500
    # Number of batches to use for each validation (useful if validation set is large)
    nbatches_per_validation: int = 300


def _init_model_from_config(
    feature_extractor: ResNetFeatureExtractor, model_config: ModelConfig
) -> Pose2p5DModel:
    model = Pose2p5DModel(
        n_keypoints=model_config.n_keypoints,
        feature_extractor=feature_extractor,
        depth_n_bins=model_config.depth_n_bins,
        depth_min=model_config.depth_min,
        depth_max=model_config.depth_max,
        xy_temperature=model_config.xy_temperature,
        depth_temperature=model_config.depth_temperature,
        upsample_n_layers=model_config.upsample_n_layers,
        upsample_n_hidden_channels=model_config.upsample_n_hidden_channels,
        depth_n_hidden_channels=model_config.depth_n_hidden_channels,
        confidence_method=model_config.confidence_method,
        groupnorm_n_groups=model_config.groupnorm_n_groups,
        pose_head_init_std=model_config.pose_head_init_std,
    )
    logging.info("Created 2.5D pose estimation model")
    return model


def _setup_model(model_config: ModelConfig) -> Pose2p5DModel:
    # If full model weights are provided, load them directly
    if model_config.model_weights is not None:
        checkpoint_path = Path(model_config.model_weights)
        if not checkpoint_path.is_file():
            raise ValueError(f"Model weights path {checkpoint_path} is not a file")
        weights = torch.load(checkpoint_path, map_location="cpu")
        feature_extractor = ResNetFeatureExtractor(weights=None)
        model = _init_model_from_config(feature_extractor, model_config)
        model.load_state_dict(weights)
        logging.info(f"Set up model using weights from {checkpoint_path}")
        return model

    # Otherwise, init feature extractor first
    fe_weights_option = model_config.feature_extractor_weights
    if fe_weights_option == "IMAGENET1K_V1":
        feature_extractor = ResNetFeatureExtractor(weights=fe_weights_option)
        logging.info(f"Created feature extractor using {fe_weights_option} weights")
    else:
        # Weights must be given as a checkpoint path then
        checkpoint_path = Path(fe_weights_option)
        if not checkpoint_path.is_file():
            raise ValueError(
                f"Feature extractor weights {fe_weights_option} is neither "
                '"IMAGENET1K_V1" nor a valid path'
            )
        feature_extractor = ResNetFeatureExtractor(weights=None)
        weights = torch.load(checkpoint_path, map_location="cpu")
        feature_extractor.load_state_dict(weights)
        logging.info(f"Created feature extractor using weights from {checkpoint_path}")
    model = _init_model_from_config(feature_extractor, model_config)
    logging.info("Set up model, incorporating feature extractor above")
    return model


def _setup_loss_func(loss_config: LossConfig) -> Pose2p5DLoss:
    loss_func = Pose2p5DLoss(**loss_config.__dict__)
    logging.info("Set up 2.5D pose estimation loss function")
    return loss_func


def train_keypoints3d_model(
    sampling: SamplingConfig,
    model: ModelConfig,
    loss: LossConfig,
    data: DataConfig,
    training: TrainingConfig,
    output: OutputConfig,
) -> None:
    """Train the 3D keypoint detection model using a pretrained feature extractor."""
    # System setup
    set_random_seed(training.seed)
    hardware_avail = get_hardware_availability(check_gpu=True, print_results=True)
    if len(hardware_avail["gpus"]) == 0:
        raise RuntimeError("No GPU available for training")
    torch.backends.cudnn.benchmark = True

    # Initialize datasets and dataloaders
    train_ds, train_loader = init_atomic_dataset_and_dataloader(
        data_dirs=data.train_data_dirs,
        atomic_batch_nsamples=sampling.atomic_batch_nsamples,
        atomic_batch_nvariants=sampling.atomic_batch_nvariants,
        image_size=data.image_size,
        train_batch_size=sampling.train_batch_size,
        load_keypoint_positions=True,
        num_workers=sampling.num_workers,
        num_channels=3,
        shuffle=True,
    )
    val_ds, val_loader = init_atomic_dataset_and_dataloader(
        data_dirs=data.val_data_dirs,
        atomic_batch_nsamples=sampling.atomic_batch_nsamples,
        atomic_batch_nvariants=sampling.atomic_batch_nvariants,
        image_size=data.image_size,
        train_batch_size=sampling.val_batch_size,
        load_keypoint_positions=True,
        num_workers=sampling.num_workers,
        num_channels=3,
        shuffle=False,
    )

    # Initialize model and loss function
    pose_model = _setup_model(model)  # `model` argument is actually ModelConfig
    criterion = _setup_loss_func(loss)  # `loss` is actually LossConfig

    # Initialize learning pipeline
    pipeline = Pose2p5DPipeline(
        model=pose_model,
        loss_func=criterion,
        depth_offset=data.depth_offset,
        device="cuda",
        use_float16=True,
    )

    # Train the model
    adam_optimizer_kwargs = {
        "lr": training.adam_lr,
        "weight_decay": training.adam_weight_decay,
    }
    log_dir = Path(output.output_basedir) / "logs"
    checkpoint_dir = Path(output.output_basedir) / "checkpoints"
    pipeline.train(
        training_data_loader=train_loader,
        validation_data_loader=val_loader,
        num_epochs=training.num_epochs,
        adam_kwargs=adam_optimizer_kwargs,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        log_interval=output.logging_interval,
        checkpoint_interval=output.checkpoint_interval,
        validation_interval=output.validation_interval,
        nbatches_per_validation=output.nbatches_per_validation,
    )

    logging.info("Training complete")


if __name__ == "__main__":
    import tyro

    # args = tyro.cli(
    #     train_keypoints3d_model,
    #     prog=f"python {Path(__file__).name}",
    #     description="Train a 3D keypoint detection model using pretrained feature extractor.",
    # )
    
    sampling_config = SamplingConfig(
        atomic_batch_nsamples=32,
        atomic_batch_nvariants=4,
        train_batch_size=32,  # consumes ~7 GB GPU memory, limit on my 3080 Ti
        val_batch_size=32,
        num_workers=8,
    )
    pretraining_trial_dir = Path(
        "bulk_data/pose_estimation/contrastive_pretraining/trial_20251001a/"
    )
    checkpoint_to_use = (
        pretraining_trial_dir
        / "checkpoints/checkpoint_epoch009_step003000.feature_extractor.pth"
    )
    model_config = ModelConfig(
        feature_extractor_weights=str(checkpoint_to_use),
        model_weights=None,
        projection_head_hidden_dim=512,
        projection_head_output_dim=256,
        n_keypoints=32,
        depth_n_bins=64,
        depth_min=0.0,
        depth_max=2.0,
        xy_temperature=0.8,
        depth_temperature=0.8,
        upsample_n_layers=3,
        upsample_n_hidden_channels=256,
        depth_n_hidden_channels=256,
        confidence_method="entropy",
        groupnorm_n_groups=32,
        pose_head_init_std=1e-3,
    )
    loss_config = LossConfig(
        heatmap_loss_func="kl",
        heatmap_sigma=2.0,
        depth_sigma_bins=1.0,
        xy_loss_weight=4.0,
        depth_ce_loss_weight=1.0,
        depth_l1_loss_weight=0.25,
        clamp_labels=True,
    )
    data_basedir = Path("bulk_data/pose_estimation/atomic_batches")
    train_data_dirs = [
        data_basedir / f"BO_Gal4_fly{fly}_trial{trial:03d}"
        for fly in range(1, 5)  # flies 1-4
        for trial in range(1, 6)  # trials 1-5
    ]
    val_data_dirs = [data_basedir / f"BO_Gal4_fly1_trial001"]
    data_config = DataConfig(
        train_data_dirs=[str(path) for path in train_data_dirs],
        val_data_dirs=[str(path) for path in val_data_dirs],
        image_size=(256, 256),
        depth_offset=100.0,
    )
    training_config = TrainingConfig(
        num_epochs=10,
        seed=42,
        adam_lr=3e-4,
        adam_weight_decay=1e-4,
    )
    output_config = OutputConfig(
        output_basedir="bulk_data/pose_estimation/keypoints3d/trial_20250103a/",
        logging_interval=10,
        checkpoint_interval=500,
        validation_interval=500,
        nbatches_per_validation=300,
    )
    train_keypoints3d_model(
        sampling=sampling_config,
        model=model_config,
        loss=loss_config,
        data=data_config,
        training=training_config,
        output=output_config,
    )