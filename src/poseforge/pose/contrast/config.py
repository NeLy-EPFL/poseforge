from dataclasses import dataclass

from poseforge.util import SerializableDataClass


@dataclass(frozen=True)
class ModelArchitectureConfig(SerializableDataClass):
    # Number of hidden dimensions in the contrastive projection head (3-layer MLP)
    projection_head_hidden_dim: int
    # Number of output dimensions in the contrastive projection head (3-layer MLP)
    projection_head_output_dim: int


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
    # Temperature parameter for scaling the logits in the InfoNCE loss
    info_nce_temperature: float


@dataclass(frozen=True)
class TrainingDataConfig(SerializableDataClass):
    # Paths to training data (recursively containing atomic batches)
    train_data_dirs: list[str]
    # Paths to validation data (recursively containing atomic batches)
    val_data_dirs: list[str]
    # Numbers of samples (frames) in each pre-extracted atomic batch
    atomic_batch_n_samples: int
    # Number of variants (synthetic images made by different style transfer models)
    atomic_batch_n_variants: int
    # Number of different frames to include in each batch. Note that n_variants variants
    # of each frame will be included, so effective batch size =
    # train_batch_size * n_variants.
    # This must be a multiple of `atomic_epoch_nsamples` in `AtomicBatchDataset`.
    train_batch_size: int
    # Validation batch size. Can be much smaller than train_batch_size. Must be
    # a multiple of `atomic_epoch_nsamples` in `AtomicBatchDataset`
    val_batch_size: int
    # Frame size (height, width)
    image_size: tuple[int, int]
    # Number of workers for data loading. Use number of CPU cores if None.
    n_workers: int | None = None
    # Number of channels in input images (3 to use pretrained ResNet weights)
    n_channels: int = 3


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
