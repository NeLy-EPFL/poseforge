import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path

import poseforge.pose.contrast.config as config
from poseforge.pose.common import ResNetFeatureExtractor


class ContrastivePretrainingModel(nn.Module):
    def __init__(
        self,
        feature_extractor: ResNetFeatureExtractor,
        hidden_dim: int,
        output_dim: int,
    ):
        super(ContrastivePretrainingModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.projection_head = nn.Sequential(
            nn.Linear(feature_extractor.output_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_features = self.feature_extractor(x)
        h_features_pooled = F.adaptive_avg_pool2d(h_features, (1, 1)).flatten(
            start_dim=1
        )
        z_features = self.projection_head(h_features_pooled)
        return {
            "h_features": h_features,
            "h_features_pooled": h_features_pooled,
            "z_features": z_features,
        }

    @classmethod
    def create_architecture_from_config(
        cls, architecture_config: config.ModelArchitectureConfig | Path | str
    ) -> "ContrastivePretrainingModel":
        # Load from file if config is given as a path
        if isinstance(architecture_config, (Path, str)):
            architecture_config = config.ModelArchitectureConfig.load(
                architecture_config
            )
            logging.info(f"Loaded model architecture config from {architecture_config}")
        # Initialize feature extractor (WITHOUT WEIGHTS at this step!)
        feature_extractor = ResNetFeatureExtractor()

        # Initialize model from config (WITHOUT WEIGHTS at this step!)
        obj = cls(
            feature_extractor=feature_extractor,
            hidden_dim=architecture_config.projection_head_hidden_dim,
            output_dim=architecture_config.projection_head_output_dim,
        )

        logging.info("Created ContrastivePretrainingModel from architecture config")
        return obj

    def load_weights_from_config(
        self, weights_config: config.ModelWeightsConfig | Path | str
    ):
        # Load from file if config is given as a path
        if isinstance(weights_config, (Path, str)):
            weights_config = config.ModelWeightsConfig.load(weights_config)
            logging.info(f"Loaded model weights config from {weights_config}")

        # Check if config has either feature extractor weights or full model weights
        if (
            weights_config.feature_extractor_weights is None
            and weights_config.model_weights is None
        ):
            logging.warning("weights_config contains nothing useful. No action taken.")

        # If full model weights are provided, load them directly
        if weights_config.model_weights is not None:
            checkpoint_path = Path(weights_config.model_weights)
            if not checkpoint_path.is_file():
                raise ValueError(f"Model weights path {checkpoint_path} is not a file")
            weights = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(weights)
            logging.info(
                f"Loaded ContrastivePretrainingModel weights (inc. feature extractor) "
                "from config"
            )
            return

        # Otherwise, init feature extractor first
        self.feature_extractor = ResNetFeatureExtractor(
            # Path, str, or "IMAGENET1K_V1"
            weights=weights_config.feature_extractor_weights
        )
        logging.info("Set up feature extractor from config")


def compute_alignment_metrics(
    features: torch.Tensor, n_samples: int, n_variants: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagnostics for how well a contrastive encoder has learned the
    style-invariant representation we want.

    Computes two things, both on whatever embedding space you pass in (we
    recommend the pooled feature map, since that is what downstream
    segmentation / keypoint heads consume — the projection-head output is
    what the loss directly shapes, so it always looks artificially clean):

    - Aggregate invariance ratio: mean cosine *distance* between
      embeddings of the *same* simulated frame across variants, divided by
      mean cosine distance between embeddings of *different* frames.
      Lower is better. A perfectly invariant encoder approaches 0; a
      collapsed encoder approaches 1. This is a useful headline scalar for
      comparing training runs because it captures the property the loss
      is supposed to produce in a way that the loss value itself cannot
      (the loss can drop just from pushing negatives apart, without
      positives getting any closer).
    - Variant similarity matrix of shape (n_variants, n_variants), where
      entry (i, j) is the average cosine similarity between variant i and
      variant j of the same frame, averaged over the batch. The diagonal
      is 1 by construction. A row/column that stays low compared to the
      others flags a variant the encoder cannot identify with the rest
      (either a hard style, or a corrupted variant whose images do not
      actually share pose content with the others) — i.e. a candidate to
      inspect visually.

    Args:
        features (torch.Tensor): Embeddings of shape
            (n_variants * n_samples, feature_dim), laid out so that rows
            0..n_samples-1 are variant 0, n_samples..2*n_samples-1 are
            variant 1, etc. This is the layout produced by
            collapse_batch().
        n_samples (int): Number of unique simulated frames in the batch.
        n_variants (int): Number of style variants per frame.

    Returns:
        invariance_ratio (torch.Tensor): Scalar; cosine distance ratio.
        variant_similarity_matrix (torch.Tensor): (n_variants, n_variants).
    """
    feats = F.normalize(features, dim=1)
    feats_grouped = feats.view(n_variants, n_samples, -1)

    # Variant similarity matrix: average over frames of variant_i . variant_j.
    # Diagonal is 1 by construction (cosine of a normalized vector with itself).
    variant_similarity_matrix = (
        torch.einsum("ind,jnd->ij", feats_grouped, feats_grouped) / n_samples
    )

    # Within-frame similarity: average over off-diagonal entries of the matrix.
    eye = torch.eye(n_variants, dtype=torch.bool, device=features.device)
    within_frame_sim = variant_similarity_matrix[~eye].mean()

    # Between-frame similarity: average cosine sim over all pairs that come
    # from different frames (regardless of variant).
    sim_all = feats @ feats.T
    frame_id = torch.arange(n_samples, device=features.device).repeat(n_variants)
    same_frame = frame_id[None, :] == frame_id[:, None]
    between_frame_sim = sim_all[~same_frame].mean()

    within_distance = 1.0 - within_frame_sim
    between_distance = 1.0 - between_frame_sim
    invariance_ratio = within_distance / between_distance.clamp_min(1e-8)

    return invariance_ratio, variant_similarity_matrix


class InfoNCELoss(nn.Module):
    """Compute the InfoNCE loss, treating the same frame from different
    variants as positive pairs and different frames as negative pairs.
    """

    def __init__(self, temperature: float):
        """
        Args:
            temperature (float): Temperature parameter for scaling the
                logits.
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    @classmethod
    def create_from_config(
        cls, loss_config: config.LossConfig | Path | str
    ) -> "InfoNCELoss":
        # Load from file if config is given as a path
        if isinstance(loss_config, (Path, str)):
            loss_config = config.LossConfig.load(loss_config)
            logging.info(f"Loaded model loss config from {loss_config}")

        # Initialize loss from config
        obj = cls(temperature=loss_config.info_nce_temperature)

        logging.info("Created InfoNCELoss from loss config")
        return obj

    def forward(
        self, embeddings: torch.Tensor, n_samples: int, n_variants: int
    ) -> torch.Tensor:
        """
        Args:
            embeddings (torch.Tensor): Feature matrix of shape
                (batch_size * n_variants, feature_dim) in the embedding
                space wherein mutual information is evaluated.
            n_samples (int): Number of unique frames in the batch.
            n_variants (int): Number of variants (e.g., different
                augmentations) per unique frame.

        Returns:
            torch.Tensor: InfoNCE loss as a single float value.
        """
        device = embeddings.device

        # Construct labels for binary classification
        frame_id = torch.cat(
            [torch.arange(n_samples) for i in range(n_variants)], dim=0
        )
        labels_matrix = (frame_id[None, :] == frame_id[:, None]).to(device)
        # can be commented out after testing
        assert labels_matrix.shape == (
            n_samples * n_variants,
            n_samples * n_variants,
        ), "Shape of labels matrix does not match specified n_samples and n_variants in InfoNCELoss"

        # Compute cosine similarity matrix, which is just X @ X.T after X is normalized
        # across the feature dimensions (i.e. rows of X)
        embeddings = nn.functional.normalize(embeddings, dim=1)
        sim_matrix = embeddings @ embeddings.T
        # can be commented out after testing
        assert sim_matrix.shape == (
            n_samples * n_variants,
            n_samples * n_variants,
        ), "Shape of similarity matrix does not match specified n_samples and n_variants in InfoNCELoss"

        # Discard the main diagonal: exclude self comparison (x_anchor vs. x_anchor)
        n_rows = sim_matrix.shape[0]  # should be batch_size * n_variants
        mask = torch.eye(n_rows, dtype=torch.bool).to(device)
        # labels_matrix and sim_matrix are now both of shape (n_rows, n_rows-1)
        labels_matrix = labels_matrix[~mask].view(n_rows, -1)
        sim_matrix = sim_matrix[~mask].view(n_rows, -1)

        # Select the positive and negative pairs
        # positives: (n_rows, n_variants-1) where n_rows = batch_size * n_variants
        positives = sim_matrix[labels_matrix].view(n_rows, -1)
        # negatives: (n_rows, n_rows-n_variants)
        negatives = sim_matrix[~labels_matrix].view(n_rows, -1)
        # Check shapes (can be commented out after testing)
        assert positives.shape == (n_rows, n_variants - 1)
        assert negatives.shape == (n_rows, n_rows - n_variants)

        # Concatenate positives and negatives to form logits tensor of shape
        # (n_rows, n_rows-1). The positive pairs are always in the left-most (n_variants-1) columns.
        # The loss is computed by summing probabilities over these columns, without using explicit labels.
        logits = torch.cat([positives, negatives], dim=1) / self.temperature

        # Final loss computation
        # Note: the slightly confusing form below is equivalent to the following, but
        # with better numerical stability because everything is computed in log space
        # probs = F.softmax(logits, dim=1)
        # loss_per_sample = -torch.log(probs[:, : (n_variants - 1)].sum(dim=1))
        log_probs = F.log_softmax(logits, dim=1)
        loss_per_sample = -torch.logsumexp(log_probs[:, : (n_variants - 1)], dim=1)

        loss = loss_per_sample.mean()
        return loss
