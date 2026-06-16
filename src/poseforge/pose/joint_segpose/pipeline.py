import logging
from collections import defaultdict
from datetime import datetime
from itertools import chain
from pathlib import Path
from time import time
from typing import Callable

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import poseforge.pose.joint_segpose.config as config
import poseforge.pose.keypoints3d.config as keypoints3d_config
from poseforge.pose.bodyseg.model import CombinedDiceCELoss
from poseforge.pose.data.synthetic import (
    atomic_batches_to_simple_batch,
    init_atomic_dataset_and_dataloader,
)
from poseforge.pose.joint_segpose.model import JointSegPoseModel
from poseforge.pose.keypoints3d.model import Pose2p5DLoss
from poseforge.util.ml import count_module_parameters, count_optimizer_parameters
from poseforge.util.sys import (
    check_mixed_precision_status,
    clear_memory_cache,
    set_random_seed,
)


class JointSegPosePipeline:
    """Joint training pipeline for `JointSegPoseModel`.

    The two task losses are combined as
        total_loss = weight_pose * pose_total + weight_bodyseg * bodyseg_total
    and logged under namespaced keys (``pose/...``, ``bodyseg/...``).

    The shared backbone obeys the same encoder-freeze schedule as the
    pure-pose pipeline. Checkpoints are written as a single joint payload
    so the existing single-task inference scripts work unchanged.
    """

    def __init__(
        self,
        model: JointSegPoseModel,
        pose_loss: Pose2p5DLoss | None = None,
        bodyseg_loss: CombinedDiceCELoss | None = None,
        weight_pose: float = 1.0,
        weight_bodyseg: float = 1.0,
        device: torch.device | str = "cuda",
        use_float16: bool = True,
        target_label_mapper: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.model = model.to(device)
        self.pose_loss = pose_loss.to(device) if pose_loss is not None else None
        self.bodyseg_loss = bodyseg_loss.to(device) if bodyseg_loss is not None else None
        self.weight_pose = weight_pose
        self.weight_bodyseg = weight_bodyseg
        self.device = device
        if torch.cuda.is_available() and "cuda" in str(self.device):
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"
        self.use_float16 = use_float16
        self.target_label_mapper = target_label_mapper

    def train(
        self,
        n_epochs: int,
        data_config: config.JointTrainingDataConfig,
        optimizer_config: config.JointOptimizerConfig,
        artifacts_config: keypoints3d_config.TrainingArtifactsConfig,
        seed: int = 42,
    ):
        if self.pose_loss is None or self.bodyseg_loss is None:
            raise ValueError(
                "Both pose_loss and bodyseg_loss must be provided for training"
            )

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

        # Encoder-freeze schedule: encoder is param group 0.
        freeze_encoder_n_epochs = optimizer_config.freeze_encoder_n_epochs
        encoder_target_lr = optimizer_config.learning_rate_encoder
        if freeze_encoder_n_epochs > 0:
            optimizer.param_groups[0]["lr"] = 0.0
            logging.info(
                f"Shared encoder frozen for the first {freeze_encoder_n_epochs} "
                f"epoch(s). Encoder LR will be restored to {encoder_target_lr} "
                f"at epoch {freeze_encoder_n_epochs}."
            )

        grad_scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_float16)

        self.model.train()
        for epoch_idx in range(n_epochs):
            if epoch_idx == freeze_encoder_n_epochs and freeze_encoder_n_epochs > 0:
                optimizer.param_groups[0]["lr"] = encoder_target_lr
                logging.info(
                    f"Shared encoder unfrozen at epoch {epoch_idx}. "
                    f"Encoder LR set to {encoder_target_lr}."
                )

            logging.info(
                f"Starting epoch {epoch_idx} out of {n_epochs} at {datetime.now()}"
            )
            running_loss_dict = defaultdict(lambda: 0.0)
            epoch_start_time = time()
            running_start_time = time()

            for step_idx, (atomic_batches_frames, atomic_batches_sim_data) in enumerate(
                train_loader
            ):
                frames, sim_data = atomic_batches_to_simple_batch(
                    atomic_batches_frames, atomic_batches_sim_data, device=self.device
                )

                xy_labels = sim_data["keypoint_pos"][:, :, :2]
                depth_labels = sim_data["keypoint_pos"][:, :, 2]
                target_indices = sim_data["body_seg_maps"].long()
                if self.target_label_mapper is not None:
                    target_indices = self.target_label_mapper(target_indices)
                if target_indices.shape[1:] != frames.shape[2:]:
                    raise ValueError(
                        f"Target indices shape {target_indices.shape} does not match "
                        f"input frames shape {frames.shape}"
                    )

                with torch.amp.autocast(self.device_type, enabled=self.use_float16):
                    pred = self.model(frames)
                    pose_loss_dict = self.pose_loss(
                        pred["pose"],
                        xy_labels=xy_labels,
                        depth_labels=depth_labels,
                        bin_values=self.model.pose.depth_bin_centers,
                    )
                    bodyseg_loss_dict = self.bodyseg_loss(
                        pred["bodyseg"]["logits"], target_indices
                    )
                    total_loss = (
                        self.weight_pose * pose_loss_dict["total_loss"]
                        + self.weight_bodyseg * bodyseg_loss_dict["total_loss"]
                    )

                    if epoch_idx == 0 and step_idx == 0:
                        check_mixed_precision_status(
                            self.use_float16,
                            self.device,
                            print_results=True,
                            tensors={
                                "feature_extractor_params": self.model.feature_extractor.parameters(),
                                "pose_pred_xy": pred["pose"]["pred_xy"],
                                "bodyseg_logits": pred["bodyseg"]["logits"],
                            },
                            grad_scaler=grad_scaler,
                            subtitle="Variables at start of training",
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(total_loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                # Logging — namespace each sub-loss
                running_loss_dict["total_loss"] += total_loss.item()
                for key, value in pose_loss_dict.items():
                    running_loss_dict[f"pose/{key}"] += value.item()
                for key, value in bodyseg_loss_dict.items():
                    running_loss_dict[f"bodyseg/{key}"] += value.item()

                if step_idx % artifacts_config.logging_interval == 0 and step_idx > 0:
                    avg_loss_dict = {
                        k: x / artifacts_config.logging_interval
                        for k, x in running_loss_dict.items()
                    }
                    time_now = time()
                    throughput = artifacts_config.logging_interval / (
                        time_now - running_start_time
                    )
                    running_loss_dict = defaultdict(lambda: 0.0)
                    running_start_time = time_now
                    self._update_logs_training(
                        writer,
                        epoch_index=epoch_idx,
                        within_epoch_step_idx=step_idx,
                        n_batches_per_epoch=n_batches_per_epoch,
                        avg_loss_dict=avg_loss_dict,
                        throughput=throughput,
                    )

                if (
                    step_idx % artifacts_config.validation_interval == 0
                    and step_idx > 0
                ):
                    del (
                        atomic_batches_frames,
                        atomic_batches_sim_data,
                        frames,
                        sim_data,
                        pred,
                        xy_labels,
                        depth_labels,
                        target_indices,
                    )
                    clear_memory_cache()
                    val_loss_dict = self.validate(
                        val_loader,
                        max_batches=artifacts_config.n_batches_per_validation,
                    )
                    self._update_logs_validation(
                        writer,
                        epoch_idx=epoch_idx,
                        within_epoch_step_idx=step_idx,
                        n_batches_per_epoch=n_batches_per_epoch,
                        val_loss_dict=val_loss_dict,
                    )
                    clear_memory_cache()

                if (
                    step_idx % artifacts_config.checkpoint_interval == 0
                    and step_idx > 0
                ) or (step_idx == n_batches_per_epoch - 1):
                    checkpoint_path_stem = (
                        checkpoint_dir / f"epoch{epoch_idx}_step{step_idx}"
                    )
                    self._save_checkpoint(
                        checkpoint_path_stem,
                        model=self.model,
                        pose_loss=self.pose_loss,
                        bodyseg_loss=self.bodyseg_loss,
                        optimizer=optimizer,
                        grad_scaler=grad_scaler,
                    )
                    logging.info(f"Saved checkpoint to {checkpoint_path_stem}.*.pth")

            epoch_wall_time = time() - epoch_start_time
            logging.info(
                f"Finished epoch {epoch_idx} in {epoch_wall_time:.2f} seconds."
            )

        writer.close()

    def validate(
        self, validation_data_loader: DataLoader, max_batches: int | None = None
    ) -> dict[str, float]:
        if max_batches is None:
            max_batches = len(validation_data_loader)
        if max_batches <= 0:
            raise ValueError("max_batches must be positive or None")
        if self.pose_loss is None or self.bodyseg_loss is None:
            raise ValueError(
                "Both pose_loss and bodyseg_loss must be provided for validation"
            )

        total_loss_dict = defaultdict(lambda: 0.0)
        self.model.eval()
        clear_memory_cache()

        with torch.no_grad():
            for step_idx, (atomic_batches_frames, atomic_batches_sim_data) in enumerate(
                tqdm(validation_data_loader, desc="Validation", disable=None)
            ):
                if step_idx >= max_batches:
                    break

                frames, sim_data = atomic_batches_to_simple_batch(
                    atomic_batches_frames, atomic_batches_sim_data, device=self.device
                )
                xy_labels = sim_data["keypoint_pos"][:, :, :2]
                depth_labels = sim_data["keypoint_pos"][:, :, 2]
                target_indices = sim_data["body_seg_maps"].long()
                if self.target_label_mapper is not None:
                    target_indices = self.target_label_mapper(target_indices)

                with torch.amp.autocast(self.device_type, enabled=self.use_float16):
                    pred = self.model(frames)
                    pose_loss_dict = self.pose_loss(
                        pred["pose"],
                        xy_labels=xy_labels,
                        depth_labels=depth_labels,
                        bin_values=self.model.pose.depth_bin_centers,
                    )
                    bodyseg_loss_dict = self.bodyseg_loss(
                        pred["bodyseg"]["logits"], target_indices
                    )
                    total_loss = (
                        self.weight_pose * pose_loss_dict["total_loss"]
                        + self.weight_bodyseg * bodyseg_loss_dict["total_loss"]
                    )

                total_loss_dict["total_loss"] += total_loss.item()
                for key, value in pose_loss_dict.items():
                    total_loss_dict[f"pose/{key}"] += value.item()
                for key, value in bodyseg_loss_dict.items():
                    total_loss_dict[f"bodyseg/{key}"] += value.item()

        clear_memory_cache()
        self.model.train()
        n_steps_iterated = step_idx + 1
        return {k: v / n_steps_iterated for k, v in total_loss_dict.items()}

    def _create_optimizer(
        self, optimizer_config: config.JointOptimizerConfig
    ) -> torch.optim.Optimizer:
        # Param groups: shared backbone -> pose decoder -> pose heads ->
        # bodyseg decoder -> bodyseg head. Group 0 is the shared backbone
        # so the encoder-freeze schedule applies in one place.
        params = [
            {
                "params": list(self.model.feature_extractor.parameters()),
                "lr": optimizer_config.learning_rate_encoder,
            },
            {
                "params": list(
                    chain(
                        self.model.pose.dec_layer1.parameters(),
                        self.model.pose.dec_layer2.parameters(),
                        self.model.pose.dec_layer3.parameters(),
                        self.model.pose.dec_layer4.parameters(),
                    )
                ),
                "lr": optimizer_config.learning_rate_deconv,
            },
            {
                "params": list(self.model.pose.heatmap_head.parameters()),
                "lr": optimizer_config.learning_rate_heatmap_head,
            },
            {
                "params": list(self.model.pose.depth_head.parameters()),
                "lr": optimizer_config.learning_rate_depth_head,
            },
            {
                "params": list(
                    chain(
                        self.model.bodyseg.dec_layer1.parameters(),
                        self.model.bodyseg.dec_layer2.parameters(),
                        self.model.bodyseg.dec_layer3.parameters(),
                        self.model.bodyseg.dec_layer4.parameters(),
                    )
                ),
                "lr": optimizer_config.learning_rate_deconv,
            },
            {
                "params": list(
                    chain(
                        self.model.bodyseg.final_upsampler.parameters(),
                        self.model.bodyseg.classifier.parameters(),
                    )
                ),
                "lr": optimizer_config.learning_rate_segmentation_head,
            },
        ]

        optimizer = torch.optim.AdamW(
            params, weight_decay=optimizer_config.weight_decay
        )

        # Sanity check: every model parameter should be covered exactly once.
        # `count_module_parameters` dedupes the shared backbone (via .parameters()'s
        # internal memo), so it equals the sum of unique params across all groups.
        n_params_optimizer = count_optimizer_parameters(optimizer)
        n_params_model = count_module_parameters(self.model)
        assert n_params_optimizer == n_params_model, (
            f"Number of parameters in optimizer ({n_params_optimizer}) does not match "
            f"number of parameters in model ({n_params_model})."
        )
        return optimizer

    @staticmethod
    def _init_training_dataset_and_dataloader(
        data_config: config.JointTrainingDataConfig,
    ):
        return init_atomic_dataset_and_dataloader(
            data_dirs=data_config.train_data_dirs,
            atomic_batch_n_samples=data_config.atomic_batch_n_samples,
            atomic_batch_n_variants=data_config.atomic_batch_n_variants,
            input_image_size=data_config.input_image_size,
            batch_size=data_config.train_batch_size,
            load_dof_angles=False,
            load_keypoint_positions=True,
            load_body_segment_maps=True,
            shuffle=True,
            n_workers=data_config.num_workers,
            n_channels=3,
            pin_memory=True,
            drop_last=True,
        )

    @staticmethod
    def _init_validation_dataset_and_dataloader(
        data_config: config.JointTrainingDataConfig,
    ):
        return init_atomic_dataset_and_dataloader(
            data_dirs=data_config.val_data_dirs,
            atomic_batch_n_samples=data_config.atomic_batch_n_samples,
            atomic_batch_n_variants=data_config.atomic_batch_n_variants,
            input_image_size=data_config.input_image_size,
            batch_size=data_config.val_batch_size,
            load_dof_angles=False,
            load_keypoint_positions=True,
            load_body_segment_maps=True,
            shuffle=False,
            n_workers=data_config.num_workers,
            n_channels=3,
            pin_memory=True,
            drop_last=True,
        )

    def _update_logs_training(
        self,
        writer: SummaryWriter,
        *,
        epoch_index: int,
        within_epoch_step_idx: int,
        n_batches_per_epoch: int,
        avg_loss_dict: dict[str, float],
        throughput: float,
    ) -> None:
        global_step_idx = epoch_index * n_batches_per_epoch + within_epoch_step_idx
        writer.add_scalar("train/epoch", epoch_index, global_step_idx)
        log_str = (
            f"Epoch {epoch_index}, step {within_epoch_step_idx}/{n_batches_per_epoch}, "
        )
        for key, value in avg_loss_dict.items():
            log_str += f"{key}: {value:.4f}, "
            writer.add_scalar(f"train/loss/{key}", value, global_step_idx)
        log_str += f"throughput: {throughput:.2f} batches/sec"
        logging.info(log_str)
        writer.add_scalar("train/sys/throughput", throughput, global_step_idx)

    def _update_logs_validation(
        self,
        writer: SummaryWriter,
        *,
        epoch_idx: int,
        within_epoch_step_idx: int,
        n_batches_per_epoch: int,
        val_loss_dict: dict[str, float],
    ) -> None:
        global_step_idx = epoch_idx * n_batches_per_epoch + within_epoch_step_idx
        log_str = (
            f"Validation at epoch {epoch_idx}, "
            f"step {within_epoch_step_idx}/{n_batches_per_epoch}, "
        )
        for key, value in val_loss_dict.items():
            log_str += f"{key}: {value:.4f}, "
            writer.add_scalar(f"val/loss/{key}", value, global_step_idx)
        logging.info(log_str)

    @staticmethod
    def _save_checkpoint(
        checkpoint_path_stem: Path,
        model: JointSegPoseModel,
        pose_loss: Pose2p5DLoss | None = None,
        bodyseg_loss: CombinedDiceCELoss | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> None:
        # Single joint checkpoint file with both sub-state-dicts.
        model.save_joint_checkpoint(checkpoint_path_stem.with_suffix(".model.pth"))
        if pose_loss is not None or bodyseg_loss is not None:
            torch.save(
                {
                    "pose": pose_loss.state_dict() if pose_loss is not None else None,
                    "bodyseg": (
                        bodyseg_loss.state_dict() if bodyseg_loss is not None else None
                    ),
                },
                checkpoint_path_stem.with_suffix(".loss.pth"),
            )
        if optimizer is not None:
            torch.save(
                optimizer.state_dict(),
                checkpoint_path_stem.with_suffix(".optimizer.pth"),
            )
        if grad_scaler is not None:
            torch.save(
                grad_scaler.state_dict(),
                checkpoint_path_stem.with_suffix(".grad_scaler.pth"),
            )
