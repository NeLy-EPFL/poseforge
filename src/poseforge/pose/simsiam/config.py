from dataclasses import dataclass

from poseforge.util import SerializableDataClass

# Reuse the same data / weights / artifacts configs as the contrastive
# pipeline so SimSiam and InfoNCE runs are configured identically along
# those axes — this is what makes the head-to-head comparison fair.
from poseforge.pose.contrast.config import (
    ModelWeightsConfig,
    TrainingDataConfig,
    TrainingArtifactsConfig,
    OptimizerConfig,
)


__all__ = [
    "ModelWeightsConfig",
    "TrainingDataConfig",
    "TrainingArtifactsConfig",
    "OptimizerConfig",
    "SimSiamArchitectureConfig",
]


@dataclass(frozen=True)
class SimSiamArchitectureConfig(SerializableDataClass):
    # Hidden width of the projection-head MLP (3 linear layers + BN + ReLU).
    projection_head_hidden_dim: int = 512
    # Output width of the projection head (= the embedding dimension the
    # cosine-similarity loss is evaluated in).
    projection_head_output_dim: int = 256
    # Hidden width of the predictor MLP. The predictor is the *load-bearing*
    # piece of SimSiam: removing it causes the encoder to collapse to a
    # constant. The standard recipe uses a 2-layer predictor with a
    # bottleneck that is narrower than the projection output.
    predictor_hidden_dim: int = 128
