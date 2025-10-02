import torch
import time
import logging
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
from typing import Any
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from biomechpose.pose_estimation.data import concat_atomic_batches, collapse_batch
from biomechpose.pose_estimation.keypoints_3d import Pose2p5DModel, Pose2p5DLoss
from biomechpose.util import clear_memory_cache, check_mixed_precision_status


class Pose2p5DPipeline:
    def __init__(
        self,
        model: Pose2p5DModel,
        loss_func: Pose2p5DLoss,
        depth_offset: float = 100.0,
        device: torch.device | str = "cuda",
        use_float16: bool = True,
    ):
        """
        Args:
            model (Pose2p5DModel): Model to train.
            loss_func (Pose2p5DLoss): Loss function.
            depth_offset (float): Use `depth_label - depth_offset` as the
                target for depth prediction. Practically, set this to the
                working distance of the camera (distance between camera and
                bottom of the arena). Default: 100.0 (mm).
            device (torch.device | str): Device to use for training.
            use_float16 (bool): Whether to use mixed-precision in training.
        """
        self.model = model.to(device)
        self.loss_func = loss_func.to(device)
        self.depth_offset = depth_offset
        self.device = device
        if torch.cuda.is_available() and "cuda" in str(self.device):
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"
        self.use_float16 = use_float16

    def train(
        self,
        training_data_loader: DataLoader,
        validation_data_loader: DataLoader,
        num_epochs: int,
        *,
        adam_kwargs: dict[str, Any],
        log_dir: Path | str,
        checkpoint_dir: Path | str,
        log_interval: int = 10,
        checkpoint_interval: int = 100,
        validation_interval: int = 100,
        nbatches_per_validation: int | None = None,
    ):
        # Set up logging and checkpointing
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        n_batches_per_epoch = len(training_data_loader)

        # Set up optimizer
        optimizer = self._create_optimizer(adam_kwargs)

        # Set up mixed-point training
        amp_scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_float16)
        self._check_amp_status_for_model_params(
            amp_scaler, subtitle="Model parameters before training"
        )

        # Training loop
        self.model.train()
        for epoch_idx in range(num_epochs):
            logging.info(
                f"Starting epoch {epoch_idx} out of {num_epochs} at {datetime.now()}"
            )
            running_loss_dict = defaultdict(lambda: 0.0)
            epoch_start_time = time.time()
            running_start_time = time.time()

            for step_idx, (atomic_batches_frames, atomic_batches_sim_data) in enumerate(
                training_data_loader
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
                n_variants, n_samples, _, _, _ = frames_concat.shape
                frames_collapsed, sim_data_collapsed = collapse_batch(
                    frames_concat, sim_data_concat
                )

                # Run models
                with torch.amp.autocast(self.device_type, enabled=self.use_float16):
                    pred_dict = self.model(frames_collapsed)
                    xy_labels = sim_data_collapsed["keypoint_pos"][:, :, :2]
                    depth_labels_adjusted = (
                        sim_data_collapsed["keypoint_pos"][:, :, 2] - self.depth_offset
                    )
                    loss_dict = self.loss_func(
                        pred_dict,
                        xy_labels=xy_labels,
                        depth_labels=depth_labels_adjusted,
                        bin_values=self.model.depth_bin_centers,
                    )

                    # Check if float16 is used
                    if epoch_idx == 0 and step_idx == 0:
                        self._check_amp_status_for_model_params(
                            amp_scaler, subtitle="Model parameters at start of training"
                        )
                        self._check_amp_status_during_training(
                            pred_dict,
                            loss_dict,
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

                if step_idx % log_interval == 0 and step_idx > 0:
                    avg_loss_dict = {
                        k: x / log_interval for k, x in running_loss_dict.items()
                    }
                    time_now = time.time()
                    throughput = log_interval / (time_now - running_start_time)

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
                if step_idx % validation_interval == 0 and step_idx > 0:
                    val_loss_dict = self.validate(
                        validation_data_loader, max_batches=nbatches_per_validation
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
                if (step_idx % checkpoint_interval == 0 and step_idx > 0) or (
                    step_idx == n_batches_per_epoch - 1
                ):
                    checkpoint_path = (
                        checkpoint_dir / f"epoch{epoch_idx}_step{step_idx}.pt"
                    )
                    self._save_checkpoint(
                        checkpoint_path,
                        model=self.model,
                        optimizer=optimizer,
                        grad_scaler=amp_scaler,
                    )
                    logging.info(f"Saved checkpoint to {checkpoint_path}")

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
                    depth_labels_adjusted = (
                        sim_data_collapsed["keypoint_pos"][:, :, 2] - self.depth_offset
                    )
                    loss_dict = self.loss_func(
                        pred_dict,
                        xy_labels=xy_labels,
                        depth_labels=depth_labels_adjusted,
                        bin_values=self.model.depth_bin_centers,
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
                # Adjust depth predictions by adding back the offset
                pred_dict["pred_depth_adjusted"] = (
                    pred_dict["pred_depth"] + self.depth_offset
                )
        self.model.train()
        return {k: v.to(input_device) for k, v in pred_dict.items()}

    def _create_optimizer(self, adam_kwargs: dict[str, Any]) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), **adam_kwargs)

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
    def _save_checkpoint(checkpoint_path: Path, model, optimizer, grad_scaler) -> None:
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "grad_scaler": grad_scaler.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def _check_amp_status_during_training(
        self,
        pred_dict: dict[str, torch.Tensor],
        loss_dict: dict[str, torch.Tensor],
        grad_scaler: torch.amp.GradScaler,
        subtitle: str = "Variables during training",
    ):
        tensors_to_check = {}
        for k, v in pred_dict.items():
            tensors_to_check[f"pred_{k}"] = v
        for k, v in loss_dict.items():
            tensors_to_check[f"loss_{k}"] = v

        return check_mixed_precision_status(
            self.use_float16,
            self.device,
            print_results=True,
            tensors=tensors_to_check,
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
