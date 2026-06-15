from dataclasses import dataclass, field
from pathlib import Path

import yaml

import poseforge.pose.bodyseg.config as bodyseg_config
import poseforge.pose.keypoints3d.config as keypoints3d_config
from poseforge.util import SerializableDataClass


@dataclass(frozen=True)
class JointModelArchitectureConfig(SerializableDataClass):
    """Composite architecture config for joint pose + body-segmentation training.

    Serialized as ``{"pose": {...}, "bodyseg": {...}}``. The patched
    ``ModelArchitectureConfig.load`` on each task slices its half back out, so the
    same YAML can be fed to the existing single-task inference scripts unchanged.
    """

    pose: keypoints3d_config.ModelArchitectureConfig = field(
        default_factory=keypoints3d_config.ModelArchitectureConfig
    )
    bodyseg: bodyseg_config.ModelArchitectureConfig = field(
        default_factory=bodyseg_config.ModelArchitectureConfig
    )

    @classmethod
    def load(cls, path: Path | str) -> "JointModelArchitectureConfig":
        if not Path(path).is_file():
            raise FileNotFoundError(f"File does not exist: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            pose=keypoints3d_config.ModelArchitectureConfig(**data["pose"]),
            bodyseg=bodyseg_config.ModelArchitectureConfig(**data["bodyseg"]),
        )


@dataclass(frozen=True)
class JointModelWeightsConfig(SerializableDataClass):
    # Shared (contrastively) pretrained feature extractor weights, applied
    # to the single ResNet18 backbone shared between the two heads.
    feature_extractor_weights: str | None = None
    # Optional full joint-model weights; takes precedence over
    # feature_extractor_weights when provided.
    model_weights: str | None = None


@dataclass(frozen=True)
class JointLossConfig(SerializableDataClass):
    pose: keypoints3d_config.LossConfig = field(
        default_factory=keypoints3d_config.LossConfig
    )
    bodyseg: bodyseg_config.LossConfig = field(
        default_factory=bodyseg_config.LossConfig
    )
    # Outer weights combining the two task losses
    weight_pose: float = 1.0
    weight_bodyseg: float = 1.0

    @classmethod
    def load(cls, path: Path | str) -> "JointLossConfig":
        if not Path(path).is_file():
            raise FileNotFoundError(f"File does not exist: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            pose=keypoints3d_config.LossConfig(**data["pose"]),
            bodyseg=bodyseg_config.LossConfig(**data["bodyseg"]),
            weight_pose=data.get("weight_pose", 1.0),
            weight_bodyseg=data.get("weight_bodyseg", 1.0),
        )


@dataclass(frozen=True)
class JointOptimizerConfig(SerializableDataClass):
    # Single shared backbone -> one encoder LR
    learning_rate_encoder: float = 3e-5
    # Both decoders share the same LR
    learning_rate_deconv: float = 3e-4
    # Pose heads
    learning_rate_heatmap_head: float = 3e-4
    learning_rate_depth_head: float = 3e-4
    # Bodyseg head (final upsampler + classifier)
    learning_rate_segmentation_head: float = 3e-4
    weight_decay: float = 1e-5
    # Number of epochs to keep the shared encoder frozen at the start
    # (its LR is forced to 0). Same semantics as Pose2p5DPipeline.
    freeze_encoder_n_epochs: int = 0


@dataclass(frozen=True)
class JointTrainingDataConfig(SerializableDataClass):
    train_data_dirs: list[str]
    val_data_dirs: list[str]
    input_image_size: tuple[int, int]
    atomic_batch_n_samples: int
    atomic_batch_n_variants: int
    train_batch_size: int
    val_batch_size: int
    num_workers: int | None = None
