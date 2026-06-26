import io
import torch
import logging
import numpy as np
from time import time
from itertools import chain
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import poseforge.pose.contrast.config as config
from poseforge.pose.data.synthetic import (
    concat_atomic_batches,
    collapse_batch,
    init_atomic_dataset_and_dataloader,
)
from poseforge.pose.contrast.model import (
    ContrastivePretrainingModel,
    InfoNCELoss,
    compute_alignment_metrics,
)


def _render_variant_similarity_heatmap(
    matrix: torch.Tensor,
    *,
    title: str = "Variant cosine similarity",
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "coolwarm",
) -> np.ndarray:
    """Render a V x V variant-similarity matrix as a labeled heatmap image.

    Defaults assume mean-centered cosine values (range [-1, 1], diverging
    colormap centered on 0). Returns an HWC uint8 RGB array suitable for
    SummaryWriter.add_image with dataformats="HWC". NaN cells (e.g. a
    between-frame matrix from a single-frame batch) are labeled "nan".
    """
    # Local imports to keep the module import cheap when no logging happens.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    matrix_np = matrix.detach().cpu().to(torch.float32).numpy()
    n = matrix_np.shape[0]
    cmap_obj = plt.get_cmap(cmap)
    span = (vmax - vmin) or 1.0
    fig, ax = plt.subplots(figsize=(3.5 + 0.4 * n, 3.0 + 0.4 * n), dpi=100)
    im = ax.imshow(matrix_np, vmin=vmin, vmax=vmax, cmap=cmap_obj)
    for i in range(n):
        for j in range(n):
            value = matrix_np[i, j]
            if np.isfinite(value):
                # Pick black/white text from the cell's background luminance
                # so it stays readable for any colormap.
                norm = min(max((value - vmin) / span, 0.0), 1.0)
                r, g, b, _ = cmap_obj(norm)
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                color = "white" if lum < 0.5 else "black"
                label = f"{value:.2f}"
            else:
                color = "black"
                label = "nan"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"v{i}" for i in range(n)])
    ax.set_yticklabels([f"v{i}" for i in range(n)])
    ax.set_xlabel("Variant")
    ax.set_ylabel("Variant")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))
    return img


def _log_variant_similarity_heatmaps(
    writer: SummaryWriter,
    split: str,
    within_frame_matrix: torch.Tensor,
    between_frame_matrix: torch.Tensor,
    global_step: int,
) -> None:
    """Log the within- and between-frame variant-similarity heatmaps.

    ``split`` is e.g. "Train" or "Validation". The two share a color scale
    so within vs between is directly comparable by eye: if they look the
    same the encoder is not separating frames (collapse); if within is
    clearly warmer than between, that gap is the invariance signal.
    """
    writer.add_image(
        f"VariantSimilarity/{split}/WithinFrame",
        _render_variant_similarity_heatmap(
            within_frame_matrix, title="Within-frame cosine similarity"
        ),
        global_step,
        dataformats="HWC",
    )
    writer.add_image(
        f"VariantSimilarity/{split}/BetweenFrame",
        _render_variant_similarity_heatmap(
            between_frame_matrix, title="Between-frame cosine similarity"
        ),
        global_step,
        dataformats="HWC",
    )
from poseforge.util.sys import (
    clear_memory_cache,
    check_mixed_precision_status,
    set_random_seed,
)


class ContrastivePretrainingPipeline:
    def __init__(
        self,
        contrastive_pretraining_model: ContrastivePretrainingModel,
        info_nce_loss_func: InfoNCELoss | None = None,
        device: torch.device | str = "cuda",
        use_float16: bool = True,
    ):
        self.model = contrastive_pretraining_model.to(device)
        self.loss_func = info_nce_loss_func.to(device) if info_nce_loss_func else None
        self.device = device
        if torch.cuda.is_available() and "cuda" in str(self.device):
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"
        self.use_float16 = use_float16

    def _create_optimizer(
        self, optimizer_config: config.OptimizerConfig
    ) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            chain(
                self.model.feature_extractor.parameters(),
                self.model.projection_head.parameters(),
            ),
            lr=optimizer_config.adam_lr,
            weight_decay=optimizer_config.adam_weight_decay,
        )

    def _update_logs_training(
        self,
        writer: SummaryWriter,
        *,
        epoch_idx: int,
        within_epoch_step_idx: int,
        n_batches_per_epoch: int,
        avg_loss: float,
        avg_invariance_ratio: float,
        learning_rate: float,
        throughput: float,
    ) -> None:
        logging.info(
            f"Epoch {epoch_idx}, step {within_epoch_step_idx}/{n_batches_per_epoch}), "
            f"avg loss: {avg_loss:.4f}, "
            f"invariance ratio: {avg_invariance_ratio:.4f}, "
            f"lr: {learning_rate}, "
            f"throughput: {throughput:.2f} batches/second"
        )
        global_step_idx = epoch_idx * n_batches_per_epoch + within_epoch_step_idx
        writer.add_scalar("Loss/Train", avg_loss, global_step_idx)
        writer.add_scalar(
            "Invariance/Train", avg_invariance_ratio, global_step_idx
        )
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
        avg_invariance_ratio: float,
    ) -> None:
        logging.info(
            f"Validation avg loss: {avg_loss:.4f}, "
            f"invariance ratio: {avg_invariance_ratio:.4f}"
        )
        global_step_idx = epoch_idx * n_batches_per_epoch + within_epoch_step_idx
        writer.add_scalar("Loss/Validation", avg_loss, global_step_idx)
        writer.add_scalar(
            "Invariance/Validation", avg_invariance_ratio, global_step_idx
        )

    def _save_checkpoint(self, checkpoint_path_stem: Path) -> None:
        torch.save(
            self.model.feature_extractor.state_dict(),
            checkpoint_path_stem.with_suffix(".feature_extractor.pth"),
        )
        torch.save(
            self.model.projection_head.state_dict(),
            checkpoint_path_stem.with_suffix(".projection_head.pth"),
        )

    def train(
        self,
        n_epochs: int,
        data_config: config.TrainingDataConfig,
        optimizer_config: config.OptimizerConfig,
        artifacts_config: config.TrainingArtifactsConfig,
        seed: int = 42,
    ) -> None:
        # Set random seed for reproducibility
        set_random_seed(seed)

        # Set up training and validation data
        train_ds, train_loader = self._init_training_dataset_and_dataloader(data_config)
        val_ds, val_loader = self._init_validation_dataset_and_dataloader(data_config)
        n_batches_per_epoch = len(train_loader)

        # Set up logging dir and logger
        log_dir = Path(artifacts_config.output_basedir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))

        # Set up checkpointing dir
        checkpoint_dir = Path(artifacts_config.output_basedir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set up optimizer
        optimizer = self._create_optimizer(optimizer_config)

        # Set up mixed-precision training
        grad_scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_float16)
        self._check_amp_status_for_model_params(
            grad_scaler, subtitle="Model parameters before training"
        )

        # Check if loss function is provided
        if self.loss_func is None:
            raise ValueError("Loss function must be provided for training")

        # Training loop
        self.model.train()
        for epoch_idx in range(n_epochs):
            logging.info(
                f"Starting epoch {epoch_idx} out of {n_epochs} at {datetime.now()}"
            )
            running_loss = 0.0
            running_invariance_ratio = 0.0
            latest_within_matrix: torch.Tensor | None = None
            latest_between_matrix: torch.Tensor | None = None
            epoch_start_time = time()
            running_start_time = time()
            for step_idx, (atomic_batches, _) in enumerate(train_loader):
                # Merge atomic batches into a single batch. The aligned
                # random crop has already been applied per atomic batch
                # inside the DataLoader workers (one window per frame,
                # shared across that frame's variants), so atomic_batches
                # arrives at the crop size.
                atomic_batches = atomic_batches.to(self.device, non_blocking=True)
                concatenated_batch = concat_atomic_batches(atomic_batches)
                n_variants, n_samples, _, _, _ = concatenated_batch.shape
                collapsed_batch = collapse_batch(concatenated_batch)

                # Run models
                with torch.amp.autocast(self.device_type, enabled=self.use_float16):
                    pred_dict = self.model(collapsed_batch)
                    h_features = pred_dict["h_features"]
                    h_features_pooled = pred_dict["h_features_pooled"]
                    z_features = pred_dict["z_features"]
                    loss = self.loss_func(
                        z_features, n_samples=n_samples, n_variants=n_variants
                    )

                    # Check if float16 is actually being used
                    if epoch_idx == 0 and step_idx == 0:
                        self._check_amp_status_for_model_params(
                            grad_scaler,
                            subtitle="Model parameters at start of training",
                        )
                        self._check_amp_status_during_training(
                            collapsed_batch,
                            pred_dict,
                            loss,
                            grad_scaler,
                            subtitle="Variables at start of training",
                        )

                # Backpropagate and optimize
                optimizer.zero_grad(set_to_none=True)  # set_to_none saves memory
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                # Diagnostic metric on the pooled features: invariance
                # ratio (lower is better) and per-variant similarity matrix.
                # Computed on what segmentation/keypoint heads actually
                # consume, not on the projection-head output.
                with torch.no_grad():
                    metrics = compute_alignment_metrics(
                        h_features_pooled.detach().float(),
                        n_samples=n_samples,
                        n_variants=n_variants,
                    )
                running_invariance_ratio += float(metrics.invariance_ratio)
                latest_within_matrix = metrics.within_frame_matrix.detach()
                latest_between_matrix = metrics.between_frame_matrix.detach()

                # Logging
                running_loss += loss.item()
                if step_idx % artifacts_config.logging_interval == 0 and step_idx > 0:
                    avg_loss = running_loss / artifacts_config.logging_interval
                    avg_invariance_ratio = (
                        running_invariance_ratio / artifacts_config.logging_interval
                    )
                    learning_rate = optimizer.param_groups[0]["lr"]
                    throughput = artifacts_config.logging_interval / (
                        time() - running_start_time
                    )
                    self._update_logs_training(
                        writer=writer,
                        epoch_idx=epoch_idx,
                        within_epoch_step_idx=step_idx,
                        n_batches_per_epoch=n_batches_per_epoch,
                        avg_loss=avg_loss,
                        avg_invariance_ratio=avg_invariance_ratio,
                        learning_rate=learning_rate,
                        throughput=torch.nan if step_idx == 0 else throughput,
                    )
                    running_loss = 0.0
                    running_invariance_ratio = 0.0
                    running_start_time = time()

                # Run validation
                if (
                    step_idx % artifacts_config.validation_interval == 0
                    and step_idx > 0
                ):
                    logging.info(
                        f"Running validation over the first "
                        f"{artifacts_config.n_batches_per_validation} batches in the "
                        "validation set"
                    )

                    # Free GPU memory before validation
                    del (
                        atomic_batches,
                        concatenated_batch,
                        collapsed_batch,
                        pred_dict,
                        h_features,
                        h_features_pooled,
                        z_features,
                        loss,
                    )
                    clear_memory_cache()

                    (
                        avg_val_loss,
                        avg_val_invariance_ratio,
                        val_within_matrix,
                        val_between_matrix,
                    ) = self.validate(
                        val_loader,
                        max_nbatches=artifacts_config.n_batches_per_validation,
                        data_config=data_config,
                    )

                    self._update_logs_validation(
                        writer=writer,
                        epoch_idx=epoch_idx,
                        within_epoch_step_idx=step_idx,
                        n_batches_per_epoch=n_batches_per_epoch,
                        avg_loss=avg_val_loss,
                        avg_invariance_ratio=avg_val_invariance_ratio,
                    )

                    # Log the within- and between-frame heatmaps once per run.
                    global_step_idx = (
                        epoch_idx * n_batches_per_epoch + step_idx
                    )
                    _log_variant_similarity_heatmaps(
                        writer,
                        "Validation",
                        val_within_matrix,
                        val_between_matrix,
                        global_step_idx,
                    )

                # Save checkpoint
                if (
                    step_idx % artifacts_config.checkpoint_interval == 0
                    and step_idx > 0
                ) or (step_idx == n_batches_per_epoch - 1):
                    checkpoint_path_stem = (
                        checkpoint_dir
                        / f"checkpoint_epoch{epoch_idx:03d}_step{step_idx:06d}"
                    )
                    self._save_checkpoint(checkpoint_path_stem)
                    logging.info(f"Saved checkpoint: {checkpoint_path_stem}.*.pth")

                    # Log a training heatmap snapshot at the same cadence as
                    # checkpoints. Uses the most recent batch's matrices; a
                    # single snapshot is enough to spot a variant the encoder
                    # is failing to identify with the rest.
                    if latest_within_matrix is not None:
                        global_step_idx = (
                            epoch_idx * n_batches_per_epoch + step_idx
                        )
                        _log_variant_similarity_heatmaps(
                            writer,
                            "Train",
                            latest_within_matrix,
                            latest_between_matrix,
                            global_step_idx,
                        )

            end = time()
            epoch_walltime = end - epoch_start_time
            logging.info(f"Epoch {epoch_idx} completed in {epoch_walltime:.2f} seconds")

        writer.close()

    def validate(
        self,
        validation_loader: DataLoader,
        max_nbatches: int | None = None,
        data_config: config.TrainingDataConfig | None = None,
    ) -> tuple[float, float, torch.Tensor, torch.Tensor]:
        """Use the model on the validation set and compute the average loss
        plus the diagnostic alignment metrics.

        Args:
            validation_loader: DataLoader for validation data
            max_nbatches: Maximum number of batches to use from the
                validation set. If None, use all batches.
            data_config: Training data config. Only used to read crop
                settings so validation matches the training input shape.
                A deterministic center crop is used here (vs. random crop
                in training) so validation loss is comparable across
                steps. If None, no cropping is applied.

        Returns:
            avg_validation_loss (float): Average InfoNCE loss across the
                validation batches.
            avg_invariance_ratio (float): Mean within-frame / between-frame
                cosine distance ratio on the pooled features, averaged
                across batches. Lower is better.
            avg_within_frame_matrix (torch.Tensor): (n_variants, n_variants)
                average same-frame cosine similarity between variant pairs.
                Diagonal is 1; off-diagonal cells that stay low flag styles
                the encoder cannot identify with the others.
            avg_between_frame_matrix (torch.Tensor): (n_variants, n_variants)
                average cosine similarity between variant pairs of different
                frames — the floor the within-frame matrix is judged against.
        """
        # Check if loss function is provided
        if self.loss_func is None:
            raise ValueError("Loss function must be provided for training")

        # Set models to evaluation mode
        self.model.eval()

        total_loss = 0.0
        total_invariance_ratio = 0.0
        accumulated_within_matrix: torch.Tensor | None = None
        accumulated_between_matrix: torch.Tensor | None = None
        if max_nbatches is None:
            max_nbatches = len(validation_loader)
        with torch.no_grad():
            for batch_idx, (atomic_batches, _) in tqdm(
                enumerate(validation_loader), total=max_nbatches, disable=None
            ):
                if batch_idx == max_nbatches:
                    break

                # Merge atomic batches into a single batch (same as
                # training). The aligned center crop has already been
                # applied per atomic batch inside the DataLoader workers.
                atomic_batches = atomic_batches.to(self.device, non_blocking=True)
                concatenated_batch = concat_atomic_batches(atomic_batches)
                n_variants, n_samples, _, _, _ = concatenated_batch.shape
                collapsed_batch = collapse_batch(concatenated_batch)

                # Forward pass with mixed precision
                with torch.amp.autocast(self.device_type, enabled=self.use_float16):
                    pred_dict = self.model(collapsed_batch)
                    h_features_pooled = pred_dict["h_features_pooled"]
                    z_features = pred_dict["z_features"]
                    loss = self.loss_func(
                        z_features, n_samples=n_samples, n_variants=n_variants
                    )

                # Diagnostic metric on the pooled features (what downstream
                # heads will see), accumulated across batches.
                metrics = compute_alignment_metrics(
                    h_features_pooled.float(),
                    n_samples=n_samples,
                    n_variants=n_variants,
                )
                total_loss += loss.item()
                total_invariance_ratio += float(metrics.invariance_ratio)
                if accumulated_within_matrix is None:
                    accumulated_within_matrix = metrics.within_frame_matrix
                    accumulated_between_matrix = metrics.between_frame_matrix
                else:
                    accumulated_within_matrix = (
                        accumulated_within_matrix + metrics.within_frame_matrix
                    )
                    accumulated_between_matrix = (
                        accumulated_between_matrix + metrics.between_frame_matrix
                    )

        # Compute averages
        avg_validation_loss = total_loss / max_nbatches
        avg_invariance_ratio = total_invariance_ratio / max_nbatches
        avg_within_matrix = accumulated_within_matrix / max_nbatches
        avg_between_matrix = accumulated_between_matrix / max_nbatches

        # Clean up all validation tensors at the end
        del (
            atomic_batches,
            concatenated_batch,
            collapsed_batch,
            pred_dict,
            h_features_pooled,
            z_features,
            loss,
        )
        clear_memory_cache()

        # Set models back to training mode
        self.model.train()

        return (
            avg_validation_loss,
            avg_invariance_ratio,
            avg_within_matrix,
            avg_between_matrix,
        )

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
        self.model.eval()

        # Run models
        with torch.no_grad():
            batch = batch.to(self.device, non_blocking=True)
            with torch.amp.autocast(self.device_type, enabled=self.use_float16):
                pred_dict = self.model(batch)

        return {key: tensor.to(input_device) for key, tensor in pred_dict.items()}

    @staticmethod
    def _init_training_dataset_and_dataloader(data_config: config.TrainingDataConfig):
        return init_atomic_dataset_and_dataloader(
            data_dirs=data_config.train_data_dirs,
            atomic_batch_n_samples=data_config.atomic_batch_n_samples,
            atomic_batch_n_variants=data_config.atomic_batch_n_variants,
            input_image_size=data_config.image_size,
            batch_size=data_config.train_batch_size,
            n_workers=data_config.n_workers,
            n_channels=3,
            crop_size=tuple(data_config.crop_size)
            if data_config.crop_size is not None
            else None,
            crop_border_exclude=data_config.crop_border_exclude,
            crop_mode="random",
        )

    @staticmethod
    def _init_validation_dataset_and_dataloader(data_config: config.TrainingDataConfig):
        return init_atomic_dataset_and_dataloader(
            data_dirs=data_config.val_data_dirs,
            atomic_batch_n_samples=data_config.atomic_batch_n_samples,
            atomic_batch_n_variants=data_config.atomic_batch_n_variants,
            input_image_size=data_config.image_size,
            batch_size=data_config.val_batch_size,
            n_workers=data_config.n_workers,
            n_channels=3,
            crop_size=tuple(data_config.crop_size)
            if data_config.crop_size is not None
            else None,
            crop_border_exclude=data_config.crop_border_exclude,
            crop_mode="center",
        )

    def _check_amp_status_for_model_params(
        self, grad_scaler: torch.amp.GradScaler, subtitle: str = "Model parameters"
    ):
        return check_mixed_precision_status(
            self.use_float16,
            self.device,
            print_results=True,
            tensors={
                "feature_extractor_params": self.model.feature_extractor.parameters(),
                "projection_head_params": self.model.projection_head.parameters(),
            },
            grad_scaler=grad_scaler,
            subtitle=subtitle,
        )

    def _check_amp_status_during_training(
        self,
        collapsed_batch: torch.Tensor,
        pred_dict: dict[str, torch.Tensor],
        loss: torch.Tensor,
        grad_scaler: torch.amp.GradScaler,
        subtitle: str = "Variables during training",
    ):
        return check_mixed_precision_status(
            self.use_float16,
            self.device,
            print_results=True,
            tensors={
                "collapsed_batch": collapsed_batch,
                "h_features": pred_dict["h_features"],
                "h_features_pooled": pred_dict["h_features_pooled"],
                "z_features": pred_dict["z_features"],
                "loss": loss,
            },
            grad_scaler=grad_scaler,
            subtitle=subtitle,
        )
