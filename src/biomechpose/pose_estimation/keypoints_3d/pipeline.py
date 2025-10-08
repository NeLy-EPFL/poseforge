import torch
import time
import logging
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import biomechpose.pose_estimation.keypoints_3d.config as config
from biomechpose.pose_estimation.data.synthetic import (
    init_atomic_dataset_and_dataloader,
    concat_atomic_batches,
    collapse_batch,
)
from biomechpose.pose_estimation.keypoints_3d import Pose2p5DModel, Pose2p5DLoss
from biomechpose.util import (
    clear_memory_cache,
    check_mixed_precision_status,
    set_random_seed,
)


class Pose2p5DPipeline:
    def __init__(
        self,
        model: Pose2p5DModel,
        loss_func: Pose2p5DLoss | None = None,
        device: torch.device | str = "cuda",
        use_float16: bool = True,
    ):
        """
        Args:
            model (Pose2p5DModel): Model to train.
            loss_func (Pose2p5DLoss | None): Loss function. Not required if
                performing inference only.
            device (torch.device | str): Device to use for training.
            use_float16 (bool): Whether to use mixed-precision in training.
        """
        self.model = model.to(device)
        self.loss_func = loss_func.to(device) if loss_func else None
        self.device = device
        if torch.cuda.is_available() and "cuda" in str(self.device):
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"
        self.use_float16 = use_float16

    def train(
        self,
        n_epochs: int,
        data_config: config.TrainingDataConfig,
        optimizer_config: config.OptimizerConfig,
        artifacts_config: config.TrainingArtifactsConfig,
        seed: int = 42,
    ):
        # Set seed for reproducibility
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

        # Set up mixed-point training
        amp_scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_float16)
        self._check_amp_status_for_model_params(
            amp_scaler, subtitle="Model parameters before training"
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
            running_loss_dict = defaultdict(lambda: 0.0)
            epoch_start_time = time.time()
            running_start_time = time.time()

            for step_idx, (atomic_batches_frames, atomic_batches_sim_data) in enumerate(
                train_loader
            ):
                # Merge atomic batches into a single batch
                atomic_batches_frames = atomic_batches_frames.to(
                    self.device, non_blocking=True
                )
                atomic_batches_sim_data = {
                    key: val.to(self.device, non_blocking=True)
                    for key, val in atomic_batches_sim_data.items()
                }
                frames_concat, sim_data_concat = concat_atomic_batches(
                    atomic_batches_frames, atomic_batches_sim_data
                )
                frames_collapsed, sim_data_collapsed = collapse_batch(
                    frames_concat, sim_data_concat
                )

                # Run models
                with torch.amp.autocast(self.device_type, enabled=self.use_float16):
                    pred_dict = self.model(frames_collapsed)
                    xy_labels = sim_data_collapsed["keypoint_pos"][:, :, :2]
                    depth_labels = sim_data_collapsed["keypoint_pos"][:, :, 2]
                    loss_dict = self.loss_func(
                        pred_dict,
                        xy_labels=xy_labels,
                        depth_labels=depth_labels,
                        bin_values=self.model.depth_bin_centers,  # buffered upon init
                    )

                    # Check if float16 is used
                    if epoch_idx == 0 and step_idx == 0:
                        self._check_amp_status_for_model_params(
                            amp_scaler, subtitle="Model parameters at start of training"
                        )
                        self._check_amp_status_during_training(
                            pred_dict,
                            amp_scaler,
                            subtitle="Variables at start of training",
                        )

                # Backpropagate and optimize
                optimizer.zero_grad(set_to_none=True)  # set_to_none saves memory
                amp_scaler.scale(loss_dict["total_loss"]).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()

                # Logging
                for key, loss in loss_dict.items():
                    running_loss_dict[key] += loss.item()

                if step_idx % artifacts_config.logging_interval == 0 and step_idx > 0:
                    avg_loss_dict = {
                        k: x / artifacts_config.logging_interval
                        for k, x in running_loss_dict.items()
                    }
                    time_now = time.time()
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

                # Run validation
                if (
                    step_idx % artifacts_config.validation_interval == 0
                    and step_idx > 0
                ):
                    val_loss_dict = self.validate(
                        val_loader,
                        max_batches=artifacts_config.n_batches_per_validation,
                    )
                    self.update_logs_validation(
                        writer,
                        epoch_idx=epoch_idx,
                        within_epoch_step_idx=step_idx,
                        n_batches_per_epoch=n_batches_per_epoch,
                        val_loss_dict=val_loss_dict,
                    )

                # Save checkpoint
                # (every log_interval steps and last step of each epoch)
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
                        loss=self.loss_func,
                        optimizer=optimizer,
                        grad_scaler=amp_scaler,
                    )
                    logging.info(f"Saved checkpoint to {checkpoint_path_stem}.*.pth")

            epoch_wall_time = time.time() - epoch_start_time
            logging.info(
                f"Finished epoch {epoch_idx} in {epoch_wall_time:.2f} seconds."
            )

        writer.close()

    def validate(
        self, validation_data_loader: DataLoader, max_batches: int | None = None
    ):
        """Run the model on the validation set and compute the average for
        each loss term.

        Args:
            validation_data_loader (DataLoader): Validation data loader.
            max_batches (int | None): Maximum number of batches to run. If
                None, run all batches. This is handy if the validation set
                is large and you want to run a quick validation.

        Returns:
            dict[str, float]: average loss values for each loss term.
        """
        if max_batches is None:
            max_batches = len(validation_data_loader)
        if max_batches <= 0:
            raise ValueError("max_batches must be positive or None")
        total_loss_dict = defaultdict(lambda: 0.0)
        
        if self.loss_func is None:
            raise ValueError("Loss function must be provided for validation")

        self.model.eval()
        clear_memory_cache()
        with torch.no_grad():
            for step_idx, (atomic_batches_frames, atomic_batches_sim_data) in enumerate(
                tqdm(validation_data_loader, desc="Validation", disable=None)
            ):
                if step_idx >= max_batches:
                    break

                # Merge atomic batches into a single batch
                atomic_batches_frames = atomic_batches_frames.to(
                    self.device, non_blocking=True
                )
                atomic_batches_sim_data = {
                    key: val.to(self.device, non_blocking=True)
                    for key, val in atomic_batches_sim_data.items()
                }
                frames_concat, sim_data_concat = concat_atomic_batches(
                    atomic_batches_frames, atomic_batches_sim_data
                )
                frames_collapsed, sim_data_collapsed = collapse_batch(
                    frames_concat, sim_data_concat
                )

                # Run model
                with torch.amp.autocast(self.device_type, enabled=self.use_float16):
                    pred_dict = self.model(frames_collapsed)
                    xy_labels = sim_data_collapsed["keypoint_pos"][:, :, :2]
                    depth_labels_adjusted = sim_data_collapsed["keypoint_pos"][:, :, 2]
                    loss_dict = self.loss_func(
                        pred_dict,
                        xy_labels=xy_labels,
                        depth_labels=depth_labels_adjusted,
                        bin_values=self.model.depth_bin_centers,  # buffered upon init
                    )
                # Accumulate losses
                for key, loss in loss_dict.items():
                    total_loss_dict[key] += loss.item()

        clear_memory_cache()
        self.model.train()
        n_steps_iterated = step_idx + 1
        return {k: v / n_steps_iterated for k, v in total_loss_dict.items()}

    def inference(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run inference on a batch of frames. Note that this method
        expects a simple batch of frames, not atomic batches.

        Args:
            frames (torch.Tensor): Input frames of shape
                (n_batches, n_channels, n_rows, n_cols).

        Returns:
            dict[str, torch.Tensor]: Output of Pose2p5DModel.forward.
        """
        input_device = frames.device
        self.model.eval()
        with torch.no_grad():
            frames = frames.to(self.device)
            with torch.amp.autocast(self.device_type, enabled=self.use_float16):
                pred_dict = self.model(frames)
        self.model.train()
        return {
            k: v.to(input_device) if isinstance(v, torch.Tensor) else v
            for k, v in pred_dict.items()
        }

    def _create_optimizer(
        self, optimizer_config: config.OptimizerConfig
    ) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=optimizer_config.adam_lr,
            weight_decay=optimizer_config.adam_weight_decay,
        )

    @staticmethod
    def _init_training_dataset_and_dataloader(data_config: config.TrainingDataConfig):
        return init_atomic_dataset_and_dataloader(
            data_dirs=data_config.train_data_dirs,
            atomic_batch_n_samples=data_config.atomic_batch_n_samples,
            atomic_batch_n_variants=data_config.atomic_batch_n_variants,
            input_image_size=data_config.input_image_size,
            batch_size=data_config.train_batch_size,
            load_dof_angles=False,
            load_keypoint_positions=True,
            load_body_segment_maps=False,
            shuffle=True,
            num_workers=data_config.num_workers,
            num_channels=3,
            pin_memory=True,
            drop_last=True,
        )

    @staticmethod
    def _init_validation_dataset_and_dataloader(data_config: config.TrainingDataConfig):
        return init_atomic_dataset_and_dataloader(
            data_dirs=data_config.val_data_dirs,
            atomic_batch_n_samples=data_config.atomic_batch_n_samples,
            atomic_batch_n_variants=data_config.atomic_batch_n_variants,
            input_image_size=data_config.input_image_size,
            batch_size=data_config.val_batch_size,
            load_dof_angles=False,
            load_keypoint_positions=True,
            load_body_segment_maps=False,
            shuffle=False,
            num_workers=data_config.num_workers,
            num_channels=3,
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

    def update_logs_validation(
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
        model: Pose2p5DModel,
        loss: Pose2p5DLoss | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> None:
        path = checkpoint_path_stem.with_suffix(".model.pth")
        torch.save(model.state_dict(), path)
        if loss is not None:
            path = checkpoint_path_stem.with_suffix(".loss.pth")
            torch.save(loss.state_dict(), path)
        if optimizer is not None:
            path = checkpoint_path_stem.with_suffix(".optimizer.pth")
            torch.save(optimizer.state_dict(), path)
        if grad_scaler is not None:
            path = checkpoint_path_stem.with_suffix(".grad_scaler.pth")
            torch.save(grad_scaler.state_dict(), path)

    def _check_amp_status_during_training(
        self,
        pred_dict: dict[str, torch.Tensor],
        grad_scaler: torch.amp.GradScaler,
        subtitle: str = "Variables during training",
    ):
        return check_mixed_precision_status(
            self.use_float16,
            self.device,
            print_results=True,
            tensors={
                "pred_xy_heatmaps": pred_dict["xy_heatmaps"],
                "pred_depth_logits": pred_dict["depth_logits"],
                "pred_xy": pred_dict["pred_xy"],
                "pred_depth": pred_dict["pred_depth"],
                "pred_conf_xy": pred_dict["conf_xy"],
                "pred_conf_depth": pred_dict["conf_depth"],
            },
            grad_scaler=grad_scaler,
            subtitle=subtitle,
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
                "upsampling_core_params": self.model.upsampling_core.parameters(),
                "xy_heatmap_head_params": self.model.heatmap_head.parameters(),
                "depth_head_params": self.model.depth_head.parameters(),
            },
            grad_scaler=grad_scaler,
            subtitle=subtitle,
        )
