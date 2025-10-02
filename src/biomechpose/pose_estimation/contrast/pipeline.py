import torch
import torch.nn.functional as F
import time
import logging
from tqdm import tqdm
from datetime import datetime
from typing import Any
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from biomechpose.pose_estimation.feature_extractor import ResNetFeatureExtractor
from biomechpose.pose_estimation.data import concat_atomic_batches, collapse_batch
from biomechpose.pose_estimation.contrast.model import ContrastiveProjectionHead
from biomechpose.pose_estimation.contrast.loss import info_nce_loss
from biomechpose.util import clear_memory_cache, check_mixed_precision_status


class ContrastivePretrainingPipeline:
    def __init__(
        self,
        feature_extractor: ResNetFeatureExtractor,
        projection_head: ContrastiveProjectionHead,
        device: torch.device | str = "cuda",
        use_float16: bool = True,
    ):
        self.feature_extractor = feature_extractor.to(device)
        self.projection_head = projection_head.to(device)
        self.device = device
        if torch.cuda.is_available() and "cuda" in str(self.device):
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"
        self.use_float16 = use_float16

    def _create_optimizer(self, adam_kwargs: dict[str, Any]) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            list(self.feature_extractor.parameters())
            + list(self.projection_head.parameters()),
            **adam_kwargs,
        )

    def _update_logs_training(
        self,
        writer: SummaryWriter,
        *,
        epoch_idx: int,
        within_epoch_step_idx: int,
        n_batches_per_epoch: int,
        avg_loss: float,
        learning_rate: float,
        throughput: float,
    ) -> None:
        logging.info(
            f"Epoch {epoch_idx}, step {within_epoch_step_idx}/{n_batches_per_epoch}), "
            f"avg loss: {avg_loss:.4f}, lr: {learning_rate}, "
            f"throughput: {throughput:.2f} batches/second"
        )
        global_step_idx = epoch_idx * n_batches_per_epoch + within_epoch_step_idx
        writer.add_scalar("Loss/Train", avg_loss, global_step_idx)
        writer.add_scalar("Training/Epoch", epoch_idx, global_step_idx)
        writer.add_scalar("Training/LearningRate", learning_rate, global_step_idx)
        writer.add_scalar("Training/Throughput", throughput, global_step_idx)

    def _update_logs_validation(
        self,
        writer: SummaryWriter,
        *,
        epoch_idx: int,
        within_epoch_step_idx: int,
        n_batches_per_epoch: int,
        avg_loss: float,
    ) -> None:
        logging.info(f"Validation avg loss: {avg_loss:.4f}")
        global_step_idx = epoch_idx * n_batches_per_epoch + within_epoch_step_idx
        writer.add_scalar("Loss/Validation", avg_loss, global_step_idx)

    def _save_checkpoint(self, checkpoint_path_stem: Path) -> None:
        torch.save(
            self.feature_extractor.state_dict(),
            checkpoint_path_stem.with_suffix(".feature_extractor.pth"),
        )
        torch.save(
            self.projection_head.state_dict(),
            checkpoint_path_stem.with_suffix(".projection_head.pth"),
        )

    def train(
        self,
        training_data_loader: DataLoader,
        validation_data_loader: DataLoader,
        num_epochs: int,
        *,
        temperature: float = 0.1,
        adam_kwargs: dict[str, Any],
        log_dir: Path | str,
        checkpoint_dir: Path | str,
        log_interval: int = 10,
        checkpoint_interval: int = 100,
        validation_interval: int = 100,
        nbatches_per_validation: int | None = None,
    ) -> None:
        # Set up logging and checkpointing
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        n_batches_per_epoch = len(training_data_loader)

        # Set up optimizer
        optimizer = self._create_optimizer(adam_kwargs)

        # Set up mixed-precision training
        amp_scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_float16)
        check_mixed_precision_status(
            self.use_float16,
            self.device,
            print_results=True,
            tensors={
                "feature_extractor_params": self.feature_extractor.parameters(),
                "projection_head_params": self.projection_head.parameters(),
            },
            grad_scaler=amp_scaler,
            subtitle="Initial model parameter dtypes",
        )

        # Training loop
        for epoch_idx in range(num_epochs):
            logging.info(
                f"Starting epoch {epoch_idx} out of {num_epochs} at {datetime.now()}"
            )
            running_loss = 0.0
            epoch_start_time = time.time()
            running_start_time = time.time()
            for step_idx, (atomic_batches, _) in enumerate(training_data_loader):
                # Merge atomic batches into a single batch
                atomic_batches = atomic_batches.to(self.device, non_blocking=True)
                concatenated_batch = concat_atomic_batches(atomic_batches)
                n_variants, n_samples, _, _, _ = concatenated_batch.shape
                collapsed_batch = collapse_batch(concatenated_batch)

                # Run models
                with torch.amp.autocast(self.device_type, enabled=self.use_float16):
                    h_features = self.feature_extractor(collapsed_batch)
                    h_features_pooled = F.adaptive_avg_pool2d(
                        h_features, (1, 1)
                    ).flatten(start_dim=1)
                    z_features = self.projection_head(h_features_pooled)
                    loss = info_nce_loss(
                        z_features,
                        temperature,
                        n_samples=n_samples,
                        n_variants=n_variants,
                        device=self.device,
                    )

                    # Check if float16 is actually being used
                    if epoch_idx == 0 and step_idx == 0:
                        check_mixed_precision_status(
                            self.use_float16,
                            self.device,
                            tensors={
                                "collapsed_batch": collapsed_batch,
                                "h_features": h_features,
                                "h_features_pooled": h_features_pooled,
                                "z_features": z_features,
                                "loss": loss,
                            },
                            grad_scaler=amp_scaler,
                            print_results=True,
                            subtitle="First training step",
                        )

                # Backpropagate and optimize
                optimizer.zero_grad(set_to_none=True)  # set_to_none saves memory
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()

                # Logging
                running_loss += loss.item()
                if step_idx % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    learning_rate = optimizer.param_groups[0]["lr"]
                    throughput = log_interval / (time.time() - running_start_time)
                    self._update_logs_training(
                        writer=writer,
                        epoch_idx=epoch_idx,
                        within_epoch_step_idx=step_idx,
                        n_batches_per_epoch=n_batches_per_epoch,
                        avg_loss=avg_loss,
                        learning_rate=learning_rate,
                        throughput=torch.nan if step_idx == 0 else throughput,
                    )
                    running_loss = 0.0
                    running_start_time = time.time()

                # Save checkpoint
                if step_idx % checkpoint_interval == 0:
                    checkpoint_path_stem = (
                        checkpoint_dir
                        / f"checkpoint_epoch{epoch_idx:03d}_step{step_idx:06d}"
                    )
                    self._save_checkpoint(checkpoint_path_stem)
                    logging.info(f"Saved checkpoint: {checkpoint_path_stem}.*.pth")

                # Run validation
                if step_idx % validation_interval == 0 and step_idx > 0:
                    logging.info(
                        f"Running validation over the first {nbatches_per_validation} "
                        "batches in the validation set"
                    )

                    # Free GPU memory before validation
                    # Delete training batch tensors
                    del (
                        atomic_batches,
                        concatenated_batch,
                        collapsed_batch,
                        h_features,
                        h_features_pooled,
                        z_features,
                        loss,
                    )
                    clear_memory_cache()

                    avg_val_loss = self.validate(
                        validation_data_loader,
                        temperature=temperature,
                        max_nbatches=nbatches_per_validation,
                    )

                    self._update_logs_validation(
                        writer=writer,
                        epoch_idx=epoch_idx,
                        within_epoch_step_idx=step_idx,
                        n_batches_per_epoch=n_batches_per_epoch,
                        avg_loss=avg_val_loss,
                    )

            end = time.time()
            epoch_walltime = end - epoch_start_time
            logging.info(f"Epoch {epoch_idx} completed in {epoch_walltime:.2f} seconds")

        writer.close()

    def validate(
        self,
        validation_loader: DataLoader,
        temperature: float = 0.1,
        max_nbatches: int | None = None,
    ) -> float:
        """Use the model on the validation set and compute the average loss.

        Args:
            validation_loader: DataLoader for validation data
            temperature: Temperature parameter for InfoNCE loss
            max_nbatches: Maximum number of batches to use from the
                validation set. If None, use all batches.

        Returns:
            Average validation loss
        """
        # Set models to evaluation mode
        self.feature_extractor.eval()
        self.projection_head.eval()

        total_loss = 0.0
        if max_nbatches is None:
            max_nbatches = len(validation_loader)
        with torch.no_grad():
            for batch_idx, (atomic_batches, _) in tqdm(
                enumerate(validation_loader), total=max_nbatches, disable=None
            ):
                if batch_idx == max_nbatches:
                    break

                # Merge atomic batches into a single batch (same as training)
                atomic_batches = atomic_batches.to(self.device, non_blocking=True)
                concatenated_batch = concat_atomic_batches(atomic_batches)
                n_variants, n_samples, _, _, _ = concatenated_batch.shape
                collapsed_batch = collapse_batch(concatenated_batch)

                # Forward pass with mixed precision
                with torch.amp.autocast(self.device_type, enabled=self.use_float16):
                    h_features = self.feature_extractor(collapsed_batch)
                    h_features_pooled = F.adaptive_avg_pool2d(
                        h_features, (1, 1)
                    ).flatten(start_dim=1)
                    z_features = self.projection_head(h_features_pooled)
                    loss = info_nce_loss(
                        z_features,
                        temperature,
                        n_samples=n_samples,
                        n_variants=n_variants,
                        device=self.device,
                    )

                total_loss += loss.item()

        # Compute average validation loss
        avg_validation_loss = total_loss / max_nbatches

        # Clean up all validation tensors at the end
        del (
            atomic_batches,
            concatenated_batch,
            collapsed_batch,
            h_features,
            z_features,
            loss,
        )
        clear_memory_cache()

        # Set models back to training mode
        self.feature_extractor.train()
        self.projection_head.train()

        return avg_validation_loss

    def inference(self, batch: torch.Tensor) -> torch.Tensor:
        """Run the feature extractor and projection head on a batch of
        images, agnostic to which simulated frames they come from and which
        style transfer model is used.

        Args:
            batch (torch.Tensor): Batch of shape (batch_size, C, H, W)

        Returns:
            h_features (torch.Tensor): Features from the feature extractor
            h_features_pooled (torch.Tensor): Globally pooled features from
                the feature extractor.
            z_features (torch.Tensor): Features from the projection head
        """
        # batch: (batch_size, C, H, W)
        input_device = batch.device
        if input_device != self.device:
            batch = batch.to(self.device, non_blocking=True)

        # Set models to evaluation mode
        self.feature_extractor.eval()
        self.projection_head.eval()

        # Run models
        with torch.no_grad():
            batch = batch.to(self.device, non_blocking=True)
            with torch.amp.autocast(self.device_type, enabled=self.use_float16):
                h_features = self.feature_extractor(batch)
                h_features_pooled = F.adaptive_avg_pool2d(h_features, (1, 1)).flatten(
                    start_dim=1
                )
                z_features = self.projection_head(h_features_pooled)

        return (
            h_features.to(input_device),
            h_features_pooled.to(input_device),
            z_features.to(input_device),
        )
