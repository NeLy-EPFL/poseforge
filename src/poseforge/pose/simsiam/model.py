import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path

import poseforge.pose.simsiam.config as config
from poseforge.pose.common import ResNetFeatureExtractor


class SimSiamProjectionHead(nn.Module):
    """Three-layer MLP with BatchNorm, as in the SimSiam paper. The final
    BatchNorm (without affine on the output, in the original recipe) is
    important — it stabilizes the representation and is part of the
    anti-collapse machinery."""

    def __init__(self, in_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SimSiamPredictor(nn.Module):
    """Two-layer MLP predictor sitting on top of the projection. Operates
    on one side of each pair only; the other side is the stop-gradient
    target. Removing this module makes the encoder collapse."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SimSiamPretrainingModel(nn.Module):
    """SimSiam: a single shared encoder + projection head, plus a
    predictor that sits on top of the projection on ONE side of each
    pair. Both sides of a positive pair pass through the same encoder
    and projection; the asymmetry comes from the predictor and the
    stop-gradient inside the loss.

    The student feature extractor is saved with the same filename as
    the contrastive pipeline (`feature_extractor.pth`), so downstream
    segmentation / keypoint scripts pick it up without modification.
    """

    def __init__(
        self,
        feature_extractor: ResNetFeatureExtractor,
        hidden_dim: int,
        output_dim: int,
        predictor_hidden_dim: int,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.projection_head = SimSiamProjectionHead(
            in_dim=feature_extractor.output_channels,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.predictor = SimSiamPredictor(
            dim=output_dim, hidden_dim=predictor_hidden_dim
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h_features = self.feature_extractor(x)
        h_features_pooled = F.adaptive_avg_pool2d(h_features, (1, 1)).flatten(
            start_dim=1
        )
        z_features = self.projection_head(h_features_pooled)
        p_features = self.predictor(z_features)
        return {
            "h_features": h_features,
            "h_features_pooled": h_features_pooled,
            "z_features": z_features,
            "p_features": p_features,
        }

    @classmethod
    def create_architecture_from_config(
        cls,
        architecture_config: config.SimSiamArchitectureConfig | Path | str,
    ) -> "SimSiamPretrainingModel":
        if isinstance(architecture_config, (Path, str)):
            architecture_config = config.SimSiamArchitectureConfig.load(
                architecture_config
            )
            logging.info(f"Loaded SimSiam architecture config from {architecture_config}")
        feature_extractor = ResNetFeatureExtractor()
        obj = cls(
            feature_extractor=feature_extractor,
            hidden_dim=architecture_config.projection_head_hidden_dim,
            output_dim=architecture_config.projection_head_output_dim,
            predictor_hidden_dim=architecture_config.predictor_hidden_dim,
        )
        logging.info("Created SimSiamPretrainingModel from architecture config")
        return obj

    def load_weights_from_config(
        self, weights_config: config.ModelWeightsConfig | Path | str
    ) -> None:
        if isinstance(weights_config, (Path, str)):
            weights_config = config.ModelWeightsConfig.load(weights_config)
            logging.info(f"Loaded model weights config from {weights_config}")

        if (
            weights_config.feature_extractor_weights is None
            and weights_config.model_weights is None
        ):
            logging.warning("weights_config contains nothing useful. No action taken.")

        if weights_config.model_weights is not None:
            checkpoint_path = Path(weights_config.model_weights)
            if not checkpoint_path.is_file():
                raise ValueError(f"Model weights path {checkpoint_path} is not a file")
            weights = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(weights)
            logging.info("Loaded full SimSiamPretrainingModel weights from config")
            return

        self.feature_extractor = ResNetFeatureExtractor(
            weights=weights_config.feature_extractor_weights
        )
        logging.info("Set up feature extractor from config")


class SimSiamLoss(nn.Module):
    """Negative cosine similarity between the predictor output on one
    side and the (stop-gradient) projection output on the other,
    averaged over all ordered (i, j) variant pairs with i != j. The
    symmetric form (both directions for every pair) is built-in: when
    you average over ordered pairs you cover (i->j) and (j->i) equally.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def create(cls) -> "SimSiamLoss":
        return cls()

    def forward(
        self,
        predictor_features: torch.Tensor,
        projection_features: torch.Tensor,
        n_samples: int,
        n_variants: int,
    ) -> torch.Tensor:
        """
        Args:
            predictor_features: (n_variants * n_samples, dim) — predictor
                outputs for every variant.
            projection_features: (n_variants * n_samples, dim) — raw
                projection outputs. Will be detached here so no gradient
                flows through the target side.
            n_samples: unique frames per batch.
            n_variants: variants per frame.
        """
        p = predictor_features.view(n_variants, n_samples, -1)
        # stop-gradient on the target side is the core anti-collapse
        # mechanism of SimSiam; without it the model trivially collapses
        # to a constant within a few hundred steps.
        z = projection_features.detach().view(n_variants, n_samples, -1)

        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)

        total_loss = predictor_features.new_zeros(())
        n_pairs = 0
        for i in range(n_variants):
            for j in range(n_variants):
                if i == j:
                    continue
                # Negative cosine similarity averaged across the batch.
                neg_cos = -(p[i] * z[j]).sum(dim=-1).mean()
                total_loss = total_loss + neg_cos
                n_pairs += 1

        return total_loss / max(n_pairs, 1)
