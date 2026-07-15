import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from dataclasses import dataclass
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
        feature_extractor = ResNetFeatureExtractor(backbone=architecture_config.backbone)

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

        # Otherwise, init feature extractor first, keeping the backbone that was
        # already selected by create_architecture_from_config.
        self.feature_extractor = ResNetFeatureExtractor(
            # Path, str, or "IMAGENET1K_V1"
            weights=weights_config.feature_extractor_weights,
            backbone=self.feature_extractor.backbone,
        )
        logging.info("Set up feature extractor from config")


@dataclass
class AlignmentMetrics:
    """Diagnostics returned by :func:`compute_alignment_metrics`.

    All quantities are computed on mean-centered, L2-normalized features
    (see the function docstring for why centering matters).

    Attributes:
        invariance_ratio (torch.Tensor): Scalar. Within-frame cosine
            *distance* divided by between-frame cosine distance. Lower is
            better; ~0 = style-invariant, ~1 = collapsed / no more aligned
            than two unrelated frames.
        within_frame_sim (torch.Tensor): Scalar. Mean off-diagonal of
            ``within_frame_matrix`` (same frame, different styles).
        between_frame_sim (torch.Tensor): Scalar. Mean of
            ``between_frame_matrix`` (different frames). The "floor" that
            within-frame similarity is judged against. NaN if n_samples < 2.
        within_frame_matrix (torch.Tensor): (n_variants, n_variants). Entry
            (i, j) is the mean cosine similarity between variant i and
            variant j of the *same* frame. Diagonal is 1 by construction.
        between_frame_matrix (torch.Tensor): (n_variants, n_variants). Entry
            (i, j) is the mean cosine similarity between variant i and
            variant j of *different* frames. All-NaN if n_samples < 2.
    """

    invariance_ratio: torch.Tensor
    within_frame_sim: torch.Tensor
    between_frame_sim: torch.Tensor
    within_frame_matrix: torch.Tensor
    between_frame_matrix: torch.Tensor


def compute_alignment_metrics(
    features: torch.Tensor, n_samples: int, n_variants: int
) -> AlignmentMetrics:
    """Diagnostics for how well a contrastive encoder has learned the
    style-invariant representation we want.

    All metrics are computed on whatever embedding space you pass in (we
    recommend the pooled feature map, since that is what downstream
    segmentation / keypoint heads consume — the projection-head output is
    what the loss directly shapes, so it always looks artificially clean).

    Features are **mean-centered** (the batch-mean embedding is subtracted)
    before being L2-normalized. Pooled CNN features are strongly anisotropic
    — they live in a narrow cone, so the cosine similarity of *every* pair
    (same frame or not) is crushed into a narrow band near 1 and absolute
    values are uninformative. Subtracting the common mean direction removes
    that shared component and lets the genuine structure spread out across
    the full [-1, 1] range, which makes both the matrices and the invariance
    ratio actually discriminative between runs.

    Two V x V matrices are returned, both useful to log side by side:

    - ``within_frame_matrix``: entry (i, j) = mean cosine similarity between
      variant i and variant j of the *same* frame. Diagonal is 1 by
      construction. A row/column that stays low flags a style the encoder
      cannot identify with the rest.
    - ``between_frame_matrix``: entry (i, j) = mean cosine similarity between
      variant i and variant j of *different* frames — the similarity floor.
      If the within- and between-frame matrices look the same, the encoder
      is not separating frames (collapse); if within is clearly higher than
      between, that gap is the invariance signal.

    Args:
        features (torch.Tensor): Embeddings of shape
            (n_variants * n_samples, feature_dim), laid out so that rows
            0..n_samples-1 are variant 0, n_samples..2*n_samples-1 are
            variant 1, etc. This is the layout produced by
            collapse_batch().
        n_samples (int): Number of unique simulated frames in the batch.
        n_variants (int): Number of style variants per frame.

    Returns:
        AlignmentMetrics: invariance ratio, within/between scalar
            similarities, and the within/between V x V matrices.
    """
    # Mean-center to remove the anisotropic common component, then normalize.
    feats = features - features.mean(dim=0, keepdim=True)
    feats = F.normalize(feats, dim=1)
    feats_grouped = feats.view(n_variants, n_samples, -1)

    # within_sum[i, j] = sum over frames n of <variant_i(n), variant_j(n)>.
    within_sum = torch.einsum("ind,jnd->ij", feats_grouped, feats_grouped)
    within_frame_matrix = within_sum / n_samples  # diagonal == 1 by construction

    # all_pairs_sum[i, j] = sum over ALL frame pairs (m, n) of
    # <variant_i(m), variant_j(n)> = <sum_m variant_i(m), sum_n variant_j(n)>.
    # Subtracting the m == n terms (within_sum) leaves only m != n pairs.
    frame_sum = feats_grouped.sum(dim=1)  # (n_variants, feature_dim)
    all_pairs_sum = frame_sum @ frame_sum.T
    between_sum = all_pairs_sum - within_sum
    n_between_pairs = n_samples * (n_samples - 1)
    if n_between_pairs > 0:
        between_frame_matrix = between_sum / n_between_pairs
    else:
        # Only one frame in the batch: between-frame similarity is undefined.
        between_frame_matrix = torch.full_like(within_frame_matrix, float("nan"))

    # Scalar summaries, derived from the matrices so they stay consistent.
    eye = torch.eye(n_variants, dtype=torch.bool, device=features.device)
    within_frame_sim = within_frame_matrix[~eye].mean()
    between_frame_sim = between_frame_matrix.mean()

    within_distance = 1.0 - within_frame_sim
    between_distance = 1.0 - between_frame_sim
    invariance_ratio = within_distance / between_distance.clamp_min(1e-8)

    return AlignmentMetrics(
        invariance_ratio=invariance_ratio,
        within_frame_sim=within_frame_sim,
        between_frame_sim=between_frame_sim,
        within_frame_matrix=within_frame_matrix,
        between_frame_matrix=between_frame_matrix,
    )


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
