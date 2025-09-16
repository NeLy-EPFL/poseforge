import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from biomechpose.contrastive_pretraining.dataset import ContrastivePretrainingDataset
from biomechpose.contrastive_pretraining.model import (
    RegNetFeatureExtractor,
    ContrastiveProjectionHead,
)


class ContrastivePretrainingPipeline:
    def __init__(
        self,
        feature_extractor: RegNetFeatureExtractor,
        projection_head: ContrastiveProjectionHead,
        dataset: ContrastivePretrainingDataset,
        temperature: float = 0.1,
        device: torch.device | str = "cuda",
        use_float16: bool = True,
        seed: int = 42,
    ):
        """Initialize the contrastive pretraining pipeline.

        Args:
            feature_extractor (nn.Module): Backbone feature extractor.
            projection_head (nn.Module): Projection head to generate the
                embedding in which mutual information is evaluated.
            dataset (ContrastivePretrainingDataset): Dataset for training.
            temperature (float, optional): Temperature parameter for
                contrastive loss (i.e. scale factor before softmax).
                Defaults to 0.1.
            device (torch.device | str, optional): Device to run the model
                on. Defaults to "cuda".
            use_float16 (bool, optional): Whether to use mixed precision
                training (float16). Defaults to True.
            seed (int, optional): Random seed for reproducibility.
        """
        self.feature_extractor = feature_extractor.to(device)
        self.projection_head = projection_head.to(device)
        self.dataset = dataset
        self.temperature = temperature
        self.batch_size = dataset.batch_size
        self.n_variants = dataset.n_variants
        self.device = device
        self.use_float16 = use_float16
        self.rng = torch.Generator().manual_seed(seed)

    def info_nce_loss(self, embeddings: torch.Tensor) -> float:
        """Compute the InfoNCE loss, treating the same frame from different
        variants as positive pairs and different frames as negative pairs.

        Args:
            embeddings (torch.Tensor): Feature matrix of shape
                (batch_size * n_variants, feature_dim) in the embedding
                space wherein mutual information is evaluated.

        Returns:
            float: InfoNCE loss.
        """
        # Construct labels for binary classification
        frame_id = torch.cat(
            [torch.arange(self.batch_size) for i in range(self.n_variants)], dim=0
        )
        labels_matrix = (frame_id[None, :] == frame_id[:, None]).to(self.device)
        assert labels_matrix.shape == (  # can be commented out after testing
            self.batch_size * self.n_variants,
            self.batch_size * self.n_variants,
        )

        # Compute cosine similarity matrix, which is just X @ X.T after X is normalized
        # across the feature dimensions (i.e. rows of X)
        embeddings = nn.functional.normalize(embeddings, dim=1)
        sim_matrix = embeddings @ embeddings.T
        assert sim_matrix.shape == (  # can be commented out after testing
            self.batch_size * self.n_variants,
            self.batch_size * self.n_variants,
        )

        # Discard the main diagonal: exclude self comparison (x_anchor vs. x_anchor)
        n_rows = sim_matrix.shape[0]  # should be batch_size * n_variants
        mask = torch.eye(n_rows, dtype=torch.bool).to(self.device)
        # labels_matrix and sim_matrix are now both of shape (n_rows, n_rows-1)
        labels_matrix = labels_matrix[~mask].view(n_rows, -1)
        sim_matrix = sim_matrix[~mask].view(n_rows, -1)

        # Select the positive and negative pairs
        # positives: (n_rows, n_variants-1) where n_rows = batch_size * n_variants
        positives = sim_matrix[labels_matrix].view(n_rows, -1)
        # negatives: (n_rows, n_rows-n_variants)
        negatives = sim_matrix[~labels_matrix].view(n_rows, -1)
        # Check shapes (can be commented out after testing)
        assert positives.shape == (n_rows, self.n_variants - 1)
        assert negatives.shape == (n_rows, n_rows - self.n_variants)

        # Concatenate positives and negatives to form logits matrix of shape
        # (n_rows, n_rows-1), where logits_matrix[i, j] is the similarity/logit/log-odds
        # of samples i and j being predicted to be from the same frame.
        # The positive pair is always in the left-most column, so the correct label for
        # each row is always 0.
        logits_matrix = torch.cat([positives, negatives], dim=1) / self.temperature
        labels_index = torch.zeros(n_rows, dtype=torch.long).to(self.device)

        # Compute cross-entropy loss against the labels
        loss = F.cross_entropy(logits_matrix, labels_index, reduction="mean")
        return loss

    def train(
        self,
        n_epochs: int,
        checkpoint_dir: Path,
        checkpoint_interval_epochs: int,
        log_dir: Path,
        logging_interval_steps: int = 100,
        adam_kwargs: dict[str, Any] = {},
    ):
        """Train the model.

        Args:
            n_epochs (int): Number of training epochs.
            checkpoint_dir (Path): Directory to save model checkpoints.
            checkpoint_interval_epochs (int): Interval (in number of
                epochs) for saving model checkpoints.
            log_dir (Path): Directory to save training logs and model
                checkpoints.
            logging_interval_steps (int, optional): Interval (in number of
                steps) for logging training progress.
            adam_kwargs (dict[str, Any], optional): Additional keyword
                arguments for the Adam optimizer, which will be used for
                training.
        """
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters())
            + list(self.projection_head.parameters()),
            **adam_kwargs,
        )

        # Setup mixed precision training
        amp_scaler = torch.amp.GradScaler(self.device, enabled=self.use_float16)

        # Setup TensorBoard writer
        writer = SummaryWriter(log_dir=str(log_dir))

        for epoch_idx in range(n_epochs):
            # Randomize order of access
            indices = torch.arange(len(self.dataset))
            self.rng.shuffle(indices)
            running_loss = 0.0
            step = 0
            for sample_idx in indices:
                # Get batch of data for this iter. Note that given the special sampling
                # requirements, there's no DataLoader here. The batch is generated
                # directly by the dataset object.
                # batch: (n_variants, batch_size, C, H, W)
                batch = self.dataset[sample_idx].to(self.device)
                # collapsed_batch: flatten first 2 dims (batch_size*n_variants, C, H, W)
                collapsed_batch = batch.view(
                    self.batch_size * self.n_variants, *batch.shape[2:]
                )

                # Run models
                with torch.amp.autocast(self.device, enabled=self.use_float16):
                    # H and Z spaces as defined in the SimCLR paper
                    h_features = self.feature_extractor(collapsed_batch)
                    z_features = self.projection_head(h_features)
                    loss = self.info_nce_loss(z_features)

                # Backpropagate
                self.optimizer.zero_grad()
                amp_scaler.scale(loss).backward()
                amp_scaler.step(self.optimizer)
                amp_scaler.update()

                # Logging
                running_loss += loss.item()
                if (step + 1) % logging_interval_steps == 0:
                    avg_loss = running_loss / logging_interval_steps
                    print(
                        f"Epoch {epoch_idx} (total {n_epochs}), "
                        f"Step {step} within this epoch (total {len(self.dataset)}), "
                        f"Avg loss: {avg_loss:.4f}"
                    )
                    current_step = epoch_idx * len(self.dataset) + step
                    writer.add_scalar("Loss/Train", avg_loss.item(), current_step)
                    writer.add_scalar("Training/Epoch", epoch_idx, current_step)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    writer.add_scalar("Training/LearningRate", current_lr, current_step)
                    running_loss = 0.0
                step += 1

            # Save checkpoint
            if (epoch_idx + 1) % checkpoint_interval_epochs == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch_idx+1}.pth"
                torch.save(
                    {
                        "feature_extractor": self.feature_extractor.state_dict(),
                        "projection_head": self.projection_head.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch_idx + 1,
                    },
                    checkpoint_path,
                )
                print(f"Checkpoint saved at {checkpoint_path}")

        writer.close()
