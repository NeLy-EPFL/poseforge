from dataclasses import dataclass

from poseforge.util import SerializableDataClass


@dataclass(frozen=True)
class ModelArchitectureConfig(SerializableDataClass):
    # Number of hidden dimensions in the contrastive projection head (3-layer MLP)
    projection_head_hidden_dim: int = 512
    # Number of output dimensions in the contrastive projection head (3-layer MLP)
    projection_head_output_dim: int = 256
    # Number of body keypoints to detect
    n_keypoints: int = 32
    # Number of bins to quantize depth values (distances from camera) into
    depth_n_bins: int = 64
    # Minimum depth (distance from camera) in mm
    depth_min: float = -102.0
    # Maximum depth (distance from camera) in mm
    depth_max: float = -98.0
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


@dataclass(frozen=True)
class ModelWeightsConfig(SerializableDataClass):
    # Feature extractor weights. Can be a path to the (contrastively) pretrained weights
    # or "IMAGENET1K_V1"
    feature_extractor_weights: str | None = None
    # Model weights, optional. If provided, the model will be initialized from these
    # weights (in which case feature_extractor_weights is ignored).
    model_weights: str | None = None


@dataclass(frozen=True)
class LossConfig(SerializableDataClass):
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


@dataclass(frozen=True)
class TrainingDataConfig(SerializableDataClass):
    # Paths to training data (recursively containing atomic batches)
    train_data_dirs: list[str]
    # Paths to validation data (recursively containing atomic batches)
    val_data_dirs: list[str]
    # Frame size (height, width)
    input_image_size: tuple[int, int]
    # Numbers of samples (frames) in each pre-extracted atomic batch
    atomic_batch_n_samples: int
    # Number of variants (synthetic images made by different style transfer models)
    atomic_batch_n_variants: int
    # Number of different frames to include in each batch. Note that n_variants variants
    # of each frame will be included, so effective batch size =
    # train_batch_size * n_variants.
    # This must be a multiple of `atomic_batch_n_samples` in `AtomicBatchDataset`.
    train_batch_size: int
    # Validation batch size. Can be much smaller than train_batch_size. Must be
    # a multiple of `atomic_batch_n_samples` in `AtomicBatchDataset`
    val_batch_size: int
    # Number of workers for data loading. Use number of CPU cores if None.
    num_workers: int | None = None


@dataclass(frozen=True)
class OptimizerConfig(SerializableDataClass):
    # Learning rate for Adam optimizer
    adam_lr: float = 3e-4
    # Weight decay for Adam optimizer
    adam_weight_decay: float = 1e-4


@dataclass(frozen=True)
class TrainingArtifactsConfig(SerializableDataClass):
    # Base directory to save logs and model checkpoints
    output_basedir: str
    # Log training metrics every N steps
    logging_interval: int = 10
    # Save model checkpoint every N steps (NOT EPOCHS!)
    checkpoint_interval: int = 500
    # Run validation every N steps (NOT EPOCHS!)
    validation_interval: int = 500
    # Number of batches to use for each validation (useful if validation set is large)
    n_batches_per_validation: int = 300
