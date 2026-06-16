from dataclasses import dataclass
from pathlib import Path

import yaml

from poseforge.util import SerializableDataClass


@dataclass(frozen=True)
class ModelArchitectureConfig(SerializableDataClass):
    # Number of body keypoints to detect
    n_keypoints: int = 32
    # Number of bins to quantize depth values (distances from camera) into
    depth_n_bins: int = 64
    # Minimum depth (distance from camera) in mm
    depth_min: float = -70
    # Maximum depth (distance from camera) in mm
    depth_max: float = -63
    # Temperature param to regulate the "softness" of the predicted x-y heatmaps
    xy_temperature: float = 0.8
    # Temperature param to regulate the "softness" of the predicted depth distributions
    depth_temperature: float = 0.8
    # Number of hidden channels in each deconv layer in the core upsampling pathway
    upsample_core_out_channels: int = 64
    # Number of hidden channels in the head that predicts depth distributions
    depth_hidden_channels: int = 128
    # Method to compute confidence scores from predicted distr ("entropy" for entropy
    # over predicted distr, "peak" for highest predicted probability in the distr)
    confidence_method: str = "entropy"
    # Number of groups in GroupNorm layers
    groupnorm_n_groups: int = 32
    # Std dev for initializing of final layers in x-y and depth heads (no ReLU after)
    pose_head_init_std: float = 1e-3
    # Std dev of multiplicative Gaussian noise applied to ResNet activations during training.
    # Set to 0.0 to disable.
    activation_noise_std: float = 0.0
    # Spatial dropout probability applied between decoder layers during training.
    # Drops entire feature map channels (Dropout2d) to prevent the decoder from
    # overfitting to synthetic-specific spatial patterns. Set to 0.0 to disable.
    decoder_spatial_dropout_p: float = 0.0
    # Number of hidden layers inserted in the x-y heatmap head. 0 (default) =
    # single 3x3 conv mapping decoder features straight to per-keypoint logits
    # (backward-compatible). N > 0 inserts N (3x3 conv -> GroupNorm -> ReLU)
    # blocks at width `heatmap_hidden_channels` BEFORE the existing final 3x3
    # conv. Receptive field of the head grows by 2 heatmap pixels per added
    # hidden layer; param count grows roughly as N * 9 * C^2.
    heatmap_n_hidden_layers: int = 0
    # Width of hidden layers in the heatmap head. Only used when
    # `heatmap_n_hidden_layers > 0`. Must be a multiple of `groupnorm_n_groups`
    # and >= `groupnorm_n_groups`.
    heatmap_hidden_channels: int = 64

    @classmethod
    def load(cls, path: Path | str):
        # Joint segpose configs store {"pose": {...}, "bodyseg": {...}}.
        # Detect and pick the pose slice so existing inference paths stay unchanged.
        if not Path(path).is_file():
            raise FileNotFoundError(f"File does not exist: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and "pose" in data and "bodyseg" in data:
            data = data["pose"]
        return cls(**data)


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
    # How to handle out-of-bounds (OOB) depth labels (i.e. depth values outside
    # [depth_min, depth_max]): "clamp" (clamp to valid range), "drop" (ignore OOB
    # labels) or "ignore"
    oob_treatment: str = "drop"


@dataclass(frozen=True)
class AugmentationConfig(SerializableDataClass):
    """Configuration for training-time augmentations targeting sim-to-real
    transfer. Each augmentation can be independently enabled/disabled."""

    # --- RandomScaleCrop (scale jittering) ---
    # Enable random scale crop augmentation
    scale_crop_enabled: bool = False
    # Range of scale factors (log-uniform sampling). <1 = zoom-in, >1 = zoom-out
    scale_crop_range: tuple[float, float] = (0.7, 1.3)
    # Probability of applying scale crop per sample
    scale_crop_p: float = 0.5

    # --- DomainRandomization (photometric jitter) ---
    # Enable domain randomization augmentation
    domain_randomization_enabled: bool = False
    # Gaussian blur sigma range
    blur_sigma_range: tuple[float, float] = (0.1, 2.0)
    # Probability of applying blur per sample
    blur_p: float = 0.3
    # Downsample-then-upsample factor range (fraction of original resolution)
    downsample_factor_range: tuple[float, float] = (0.25, 0.8)
    # Probability of applying downsample-upsample per sample
    downsample_p: float = 0.3
    # Additive Gaussian noise std range (relative to [0, 1] pixel range)
    noise_std_range: tuple[float, float] = (0.0, 0.05)
    # Probability of applying sensor noise per sample
    noise_p: float = 0.4
    # Multiplicative contrast factor range
    contrast_range: tuple[float, float] = (0.6, 1.4)
    # Additive brightness shift range
    brightness_range: tuple[float, float] = (-0.1, 0.1)
    # Probability of applying contrast/brightness jitter per sample
    contrast_p: float = 0.4
    # Gamma correction exponent range
    gamma_range: tuple[float, float] = (0.7, 1.5)
    # Probability of applying gamma correction per sample
    gamma_p: float = 0.3


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
    # Learning rate for the pretrained feature extractor
    learning_rate_encoder: float = 3e-5
    # Learning rate for deconv layers in the upsampling core
    learning_rate_deconv: float = 3e-4
    # Learning rate for the x-y headmap head
    learning_rate_heatmap_head: float = 3e-4
    # Learning rate for the depth head
    learning_rate_depth_head: float = 3e-4
    # Weight decay for AdamW optimizer
    weight_decay: float = 1e-5
    # Number of epochs to keep the feature extractor (encoder) frozen at the
    # start of training. During these epochs the encoder LR is set to 0 so
    # only the decoder and heads learn. After this warm-up, the encoder LR is
    # restored to `learning_rate_encoder`. Set to 0 to disable (default).
    freeze_encoder_n_epochs: int = 0


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
