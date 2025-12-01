from dataclasses import dataclass

from poseforge.util import SerializableDataClass


@dataclass(frozen=True)
class ModelArchitectureConfig(SerializableDataClass):
    # Number of object segments.
    # Default: 6 legs * (coxa, femur, tibia, tarsus1) + thorax = 25
    n_segments: int = 25
    # Number of feature channels that are gated by a per-segment attention mechanism
    n_attention_gated_feature_channels: int = 128
    # Number of feature channels that are not gated by attention
    n_global_feature_channels: int = 128
    # Camera distance
    camera_distance: float = 100.0


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
    # Weight for the translation loss term
    translation_weight: float = 1.0
    # Weight for the rotation loss term
    rotation_weight: float = 1.0


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
    n_workers: int | None = None
    # Optional kernel size for dilating bodyseg masks
    mask_dilation_kernel: int | None = None


@dataclass(frozen=True)
class OptimizerConfig(SerializableDataClass):
    learning_rate_encoder: float = 3e-5
    learning_rate_deconv: float = 3e-4
    learning_rate_attention_heads: float = 3e-4
    learning_rate_pose6d_heads: float = 3e-4
    weight_decay: float = 1e-5


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
