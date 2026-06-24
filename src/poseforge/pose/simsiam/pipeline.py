import torch
import logging
from time import time
from itertools import chain
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import poseforge.pose.simsiam.config as config
from poseforge.pose.data.synthetic import (
    concat_atomic_batches,
    collapse_batch,
    aligned_random_crop,
    aligned_center_crop,
    init_atomic_dataset_and_dataloader,
)
from poseforge.pose.simsiam.model import SimSiamPretrainingModel, SimSiamLoss

# The alignment metric and the heatmap renderer are shared with the
# InfoNCE pipeline on purpose so the comparison is fair.
from poseforge.pose.contrast.model import compute_alignment_metrics
from poseforge.pose.contrast.pipeline import _render_variant_similarity_heatmap
from poseforge.util.sys import (
    clear_memory_cache,
    set_random_seed,
)


class SimSiamPretrainingPipeline:
    """SimSiam pretraining pipeline. Mirrors the InfoNCE pipeline in every
    respect that affects fairness — data loading, cropping, mixed
    precision, checkpointing cadence, alignment metric, heatmap logging —
    and differs only in:

    - The model has a predictor MLP on top of the projection head.
    - The loss is negative cosine similarity between predictor and a
      stop-gradient target on the other variant, averaged over all
      ordered (i, j) pairs with i != j.
    - No teacher network, no EMA, no centering or sharpening.

    The student feature extractor is saved with the same filename as
    the contrastive pipeline (`feature_extractor.pth`), so downstream
    segmentation / keypoint scripts pick it up without modification.
    """

    def __init__(
        self,
        simsiam_model: SimSiamPretrainingModel,
        simsiam_loss_func: SimSiamLoss | None = None,
        device: torch.device | str = "cuda",
        use_float16: bool = True,
    ):
        self.model = simsiam_model.to(device)
        self.loss_func = simsiam_loss_func.to(device) if simsiam_loss_func else None
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
                self.model.predictor.parameters(),
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
        avg_z_norm: float,
        learning_rate: float,
        throughput: float,
    ) -> None:
        logging.info(
            f"Epoch {epoch_idx}, step {within_epoch_step_idx}/{n_batches_per_epoch}), "
            f"avg loss: {avg_loss:.4f}, "
            f"invariance ratio: {avg_invariance_ratio:.4f}, "
            f"|z|: {avg_z_norm:.4f}, "
            f"lr: {learning_rate}, "
            f"throughput: {throughput:.2f} batches/second"
        )
        global_step_idx = epoch_idx * n_batches_per_epoch + within_epoch_step_idx
        writer.add_scalar("Loss/Train", avg_loss, global_step_idx)
        writer.add_scalar("Invariance/Train", avg_invariance_ratio, global_step_idx)
        # SimSiam-specific collapse detector: if the projection norms
        # collapse to ~0, training has degenerated to the trivial solution.
        writer.add_scalar("Diagnostics/ProjectionNorm", avg_z_norm, global_step_idx)
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
        torch.save(
            self.model.predictor.state_dict(),
            checkpoint_path_stem.with_suffix(".predictor.pth"),
        )

    def train(
        self,
        n_epochs: int,
        data_config: config.TrainingDataConfig,
        optimizer_config: config.OptimizerConfig,
        artifacts_config: config.TrainingArtifactsConfig,
        seed: int = 42,
    ) -> None:
        set_random_seed(seed)

        train_ds, train_loader = self._init_training_dataset_and_dataloader(data_config)
        val_ds, val_loader = self._init_validation_dataset_and_dataloader(data_config)
        n_batches_per_epoch = len(train_loader)

        log_dir = Path(artifacts_config.output_basedir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))

        checkpoint_dir = Path(artifacts_config.output_basedir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        optimizer = self._create_optimizer(optimizer_config)
        grad_scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_float16)

        if self.loss_func is None:
            raise ValueError("Loss function must be provided for training")

        self.model.train()
        for epoch_idx in range(n_epochs):
            logging.info(
                f"Starting epoch {epoch_idx} out of {n_epochs} at {datetime.now()}"
            )
            running_loss = 0.0
            running_invariance_ratio = 0.0
            running_z_norm = 0.0
            latest_variant_sim_matrix: torch.Tensor | None = None
            epoch_start_time = time()
            running_start_time = time()
            for step_idx, (atomic_batches, _) in enumerate(train_loader):
                atomic_batches = atomic_batches.to(self.device, non_blocking=True)
                concatenated_batch = concat_atomic_batches(atomic_batches)
                if data_config.crop_size is not None:
                    concatenated_batch = aligned_random_crop(
                        concatenated_batch,
                        crop_size=tuple(data_config.crop_size),
                        border_exclude=data_config.crop_border_exclude,
                    )
                n_variants, n_samples, _, _, _ = concatenated_batch.shape
                collapsed_batch = collapse_batch(concatenated_batch)

                with torch.amp.autocast(self.device_type, enabled=self.use_float16):
                    pred_dict = self.model(collapsed_batch)
                    h_features_pooled = pred_dict["h_features_pooled"]
                    z_features = pred_dict["z_features"]
                    p_features = pred_dict["p_features"]
                    loss = self.loss_func(
                        p_features,
                        z_features,
                        n_samples=n_samples,
                        n_variants=n_variants,
                    )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                with torch.no_grad():
                    invariance_ratio, variant_sim_matrix = compute_alignment_metrics(
                        h_features_pooled.detach().float(),
                        n_samples=n_samples,
                        n_variants=n_variants,
                    )
                    # Track |z| (pre-normalization projection norm). If
                    # this collapses to ~0 the model has degenerated.
                    z_norm = z_features.detach().float().norm(dim=-1).mean()
                running_invariance_ratio += float(invariance_ratio)
                running_z_norm += float(z_norm)
                latest_variant_sim_matrix = variant_sim_matrix.detach()

                running_loss += loss.item()
                if step_idx % artifacts_config.logging_interval == 0 and step_idx > 0:
                    avg_loss = running_loss / artifacts_config.logging_interval
                    avg_invariance_ratio = (
                        running_invariance_ratio / artifacts_config.logging_interval
                    )
                    avg_z_norm = running_z_norm / artifacts_config.logging_interval
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
                        avg_z_norm=avg_z_norm,
                        learning_rate=learning_rate,
                        throughput=torch.nan if step_idx == 0 else throughput,
                    )
                    running_loss = 0.0
                    running_invariance_ratio = 0.0
                    running_z_norm = 0.0
                    running_start_time = time()

                if (
                    step_idx % artifacts_config.validation_interval == 0
                    and step_idx > 0
                ):
                    logging.info(
                        f"Running validation over the first "
                        f"{artifacts_config.n_batches_per_validation} batches in the "
                        "validation set"
                    )
                    del (
                        atomic_batches,
                        concatenated_batch,
                        collapsed_batch,
                        pred_dict,
                        h_features_pooled,
                        z_features,
                        p_features,
                        loss,
                    )
                    clear_memory_cache()

                    (
                        avg_val_loss,
                        avg_val_invariance_ratio,
                        val_variant_sim_matrix,
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
                    global_step_idx = epoch_idx * n_batches_per_epoch + step_idx
                    heatmap_img = _render_variant_similarity_heatmap(
                        val_variant_sim_matrix
                    )
                    writer.add_image(
                        "VariantSimilarity/Validation",
                        heatmap_img,
                        global_step_idx,
                        dataformats="HWC",
                    )

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

                    if latest_variant_sim_matrix is not None:
                        global_step_idx = epoch_idx * n_batches_per_epoch + step_idx
                        heatmap_img = _render_variant_similarity_heatmap(
                            latest_variant_sim_matrix
                        )
                        writer.add_image(
                            "VariantSimilarity/Train",
                            heatmap_img,
                            global_step_idx,
                            dataformats="HWC",
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
    ) -> tuple[float, float, torch.Tensor]:
        if self.loss_func is None:
            raise ValueError("Loss function must be provided for training")

        self.model.eval()

        total_loss = 0.0
        total_invariance_ratio = 0.0
        accumulated_variant_sim_matrix: torch.Tensor | None = None
        if max_nbatches is None:
            max_nbatches = len(validation_loader)
        with torch.no_grad():
            for batch_idx, (atomic_batches, _) in tqdm(
                enumerate(validation_loader), total=max_nbatches, disable=None
            ):
                if batch_idx == max_nbatches:
                    break

                atomic_batches = atomic_batches.to(self.device, non_blocking=True)
                concatenated_batch = concat_atomic_batches(atomic_batches)
                if data_config is not None and data_config.crop_size is not None:
                    concatenated_batch = aligned_center_crop(
                        concatenated_batch,
                        crop_size=tuple(data_config.crop_size),
                    )
                n_variants, n_samples, _, _, _ = concatenated_batch.shape
                collapsed_batch = collapse_batch(concatenated_batch)

                with torch.amp.autocast(self.device_type, enabled=self.use_float16):
                    pred_dict = self.model(collapsed_batch)
                    h_features_pooled = pred_dict["h_features_pooled"]
                    z_features = pred_dict["z_features"]
                    p_features = pred_dict["p_features"]
                    loss = self.loss_func(
                        p_features,
                        z_features,
                        n_samples=n_samples,
                        n_variants=n_variants,
                    )

                invariance_ratio, variant_sim_matrix = compute_alignment_metrics(
                    h_features_pooled.float(),
                    n_samples=n_samples,
                    n_variants=n_variants,
                )
                total_loss += loss.item()
                total_invariance_ratio += float(invariance_ratio)
                if accumulated_variant_sim_matrix is None:
                    accumulated_variant_sim_matrix = variant_sim_matrix
                else:
                    accumulated_variant_sim_matrix = (
                        accumulated_variant_sim_matrix + variant_sim_matrix
                    )

        avg_validation_loss = total_loss / max_nbatches
        avg_invariance_ratio = total_invariance_ratio / max_nbatches
        avg_variant_similarity_matrix = accumulated_variant_sim_matrix / max_nbatches

        del (
            atomic_batches,
            concatenated_batch,
            collapsed_batch,
            pred_dict,
            h_features_pooled,
            z_features,
            p_features,
            loss,
        )
        clear_memory_cache()

        self.model.train()
        return (
            avg_validation_loss,
            avg_invariance_ratio,
            avg_variant_similarity_matrix,
        )

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
        )
