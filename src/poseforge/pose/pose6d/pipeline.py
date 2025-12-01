import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from collections import defaultdict
from time import time
from datetime import datetime
from pathlib import Path
from itertools import chain
from tqdm import tqdm
from loguru import logger

import poseforge.util as util
import poseforge.pose.data.synthetic as synth_data
import poseforge.pose.pose6d.config as config
from poseforge.pose.pose6d.model import Pose6DModel, Pose6DLoss
from poseforge.pose.bodyseg import BodySegmentationPipeline
from poseforge.neuromechfly.constants import segments_for_6dpose


class ComputeBodysegProbs(torch.nn.Module):
    def __init__(self, scale_factor: int):
        super(ComputeBodysegProbs, self).__init__()
        self.scale_factor = scale_factor
        self.pool = torch.nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor)

        self.pose6d_idx_to_bodyseg_idx = []
        for segment in segments_for_6dpose:
            if segment.endswith("Tarsus1"):
                segment = segment.replace("Tarsus1", "Tarsus")
            bodyseg_class_labels = list(BodySegmentationPipeline.class_labels)
            if segment not in bodyseg_class_labels:
                logger.critical(
                    f"Segment {segment} in Pose6D model has no matching segment in "
                    "body segmentation maps."
                )
                raise ValueError("Invalid segment name.")
            self.pose6d_idx_to_bodyseg_idx.append(bodyseg_class_labels.index(segment))

    def forward(
        self, atomic_batch_frames: torch.Tensor, sim_data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        segmaps = sim_data["body_seg_maps"]  # (B, H, W), value = seg idx in sim data
        bat_size, nrows, ncols = segmaps.size()

        # Convert to one-hot (B, len(pose6d_segments), H, W)
        segmaps_onehot = torch.zeros(
            (bat_size, len(segments_for_6dpose), nrows, ncols), dtype=torch.float32
        )
        for i, bodyseg_idx in enumerate(self.pose6d_idx_to_bodyseg_idx):
            segmaps_onehot[:, i, :, :] = (segmaps == bodyseg_idx).float()

        # Downsample using max pooling
        segmaps_onehot_downsampled = self.pool(segmaps_onehot)
        sim_data["segmap_probs_label"] = segmaps_onehot_downsampled.to(segmaps.device)
        sim_data.pop("body_seg_maps")  # remove original segmap

        return atomic_batch_frames, sim_data


class Pose6DPipeline:
    def __init__(
        self,
        model: Pose6DModel,
        loss_func: Pose6DLoss | None = None,
        device: torch.device | str = "cuda",
        use_float16: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.loss_func = loss_func
        self.use_float16 = use_float16
        if torch.cuda.is_available() and "cuda" in str(device):
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"

        # Body segmentation maps are at full working resolution (256x256). The Pose6D
        # model performs soft global pooling weighted by segmentation map probabilities
        # at 64x64 resolution, so we need to downsample the segmentation maps. Use max
        # pooling because it's better to slightly overestimate the presence of segments.
        self._compute_bodyseg_probs_transform = ComputeBodysegProbs(scale_factor=4)

    def train(
        self,
        n_epochs: int,
        data_config: config.TrainingDataConfig,
        optimizer_config: config.OptimizerConfig,
        artifacts_config: config.TrainingArtifactsConfig,
        seed: int = 42,
        half_batch_size_for_debugging: bool = False,
    ):
        # Set random seed for reproducibility
        util.set_random_seed(seed)

        # If half_batch_size_for_debugging, cut batch sizes in half to save memory
        self.half_batch_size_for_debugging = half_batch_size_for_debugging
        if self.half_batch_size_for_debugging:
            logger.warning(
                "Debug mode: using half batch sizes for training and validation in "
                "order to fit the model in memory for a GeForce RTX 3080 Ti."
            )

        # Set up training and validation data
        train_ds, train_loader = self._init_training_dataset_and_dataloader(data_config)
        val_ds, val_loader = self._init_validation_dataset_and_dataloader(data_config)
        n_batches_per_epoch = len(train_loader)

        # Set up logging dir and logger
        log_dir = Path(artifacts_config.output_basedir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        # Set up checkpoint dir
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
            logger.critical("Loss function must be provided for training.")
            raise ValueError("Loss function must be provided for training.")

        # Training loop
        self.model.train()
        for epoch_idx in range(n_epochs):
            logger.info(
                f"Starting epoch {epoch_idx} out of {n_epochs} at {datetime.now()}..."
            )
            running_loss_dict = defaultdict(lambda: 0.0)
            epoch_start_time = time()
            running_start_time = time()

            for step_idx, atomic_batches in enumerate(train_loader):
                # Format data
                atomic_batches_frames, atomic_batches_sim_data = atomic_batches
                frames, sim_data = synth_data.atomic_batches_to_simple_batch(
                    atomic_batches_frames, atomic_batches_sim_data, device=self.device
                )
                if self.half_batch_size_for_debugging:
                    frames, sim_data = self._get_half_batch(frames, sim_data)
                bodyseg_probs_label = sim_data["segmap_probs_label"]

                # Forward pass with mixed precision
                with torch.amp.autocast(self.device_type, enabled=self.use_float16):
                    pred_pos, pred_quat = self.model(frames, bodyseg_probs_label)
                    loss_dict = self.loss_func(
                        pred_pos, pred_quat, sim_data["mesh_pos"], sim_data["mesh_quat"]
                    )

                    # Check if float16 is used
                    if epoch_idx == 0 and step_idx == 0:
                        self._check_amp_status_for_model_params(
                            grad_scaler,
                            subtitle="Model parameters at start of training",
                        )
                        self._check_amp_status_during_training(
                            frames,
                            pred_pos,
                            pred_quat,
                            sim_data["mesh_pos"],
                            sim_data["mesh_quat"],
                            grad_scaler,
                            subtitle="Variables at start of training",
                        )

                # Backpropagate and optimize
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss_dict["total_loss"]).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                # Logging
                for key, value in loss_dict.items():
                    running_loss_dict[key] += value.item()
                if step_idx % artifacts_config.logging_interval == 0 and step_idx > 0:
                    avg_loss_dict = {
                        k: x / artifacts_config.logging_interval
                        for k, x in running_loss_dict.items()
                    }
                    time_now = time()
                    time_elapsed = time_now - running_start_time
                    throughput = artifacts_config.logging_interval / time_elapsed

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
                    del (
                        atomic_batches_frames,
                        atomic_batches_sim_data,
                        frames,
                        sim_data,
                        bodyseg_probs_label,
                        pred_pos,
                        pred_quat,
                    )
                    util.clear_memory_cache()
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
                        grad_scaler=grad_scaler,
                    )
                    logger.info(f"Saved checkpoint to {checkpoint_path_stem}.*.pth")

            epoch_wall_time = time() - epoch_start_time
            logger.info(f"Finished epoch {epoch_idx} in {epoch_wall_time:.2f} seconds.")

        writer.close()

    def validate(
        self, validation_data_loader: DataLoader, max_batches: int | None = None
    ):
        if max_batches is None:
            max_batches = len(validation_data_loader)
        if max_batches <= 0:
            raise ValueError("max_batches must be positive or None")
        if self.loss_func is None:
            raise ValueError("Loss function must be provided for validation")

        total_loss_dict = defaultdict(lambda: 0.0)
        n_steps_iterated = 0
        self.model.eval()
        with torch.no_grad():
            for step_idx, (atomic_batches_frames, atomic_batches_sim_data) in enumerate(
                tqdm(validation_data_loader, desc="Validation", disable=None)
            ):
                if step_idx >= max_batches:
                    break

                # Format data
                frames, sim_data = synth_data.atomic_batches_to_simple_batch(
                    atomic_batches_frames, atomic_batches_sim_data, device=self.device
                )
                if self.half_batch_size_for_debugging:
                    frames, sim_data = self._get_half_batch(frames, sim_data)
                bodyseg_probs_label = sim_data["segmap_probs_label"]

                # Run model
                with torch.amp.autocast(self.device_type, enabled=self.use_float16):
                    pred_pos, pred_quat = self.model(frames, bodyseg_probs_label)
                    loss_dict = self.loss_func(
                        pred_pos, pred_quat, sim_data["mesh_pos"], sim_data["mesh_quat"]
                    )

                # Accumulate losses
                for key, loss in loss_dict.items():
                    total_loss_dict[key] += loss.item()
                n_steps_iterated += 1

        del (
            atomic_batches_frames,
            atomic_batches_sim_data,
            frames,
            sim_data,
            bodyseg_probs_label,
            pred_pos,
            pred_quat,
        )
        util.clear_memory_cache()
        self.model.train()
        return {k: v / n_steps_iterated for k, v in total_loss_dict.items()}

    def inference(
        self, frames: torch.Tensor, bodyseg_probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_device = frames.device
        self.model.eval()
        with torch.no_grad():
            frames = frames.to(self.device)
            bodyseg_probs = bodyseg_probs.to(self.device)
            with torch.amp.autocast(self.device_type, enabled=self.use_float16):
                pred_pos, pred_quat = self.model(frames, bodyseg_probs)
        self.model.train()
        return pred_pos.to(input_device), pred_quat.to(input_device)

    def _init_training_dataset_and_dataloader(
        self, data_config: config.TrainingDataConfig
    ):
        return synth_data.init_atomic_dataset_and_dataloader(
            data_dirs=data_config.train_data_dirs,
            atomic_batch_n_samples=data_config.atomic_batch_n_samples,
            atomic_batch_n_variants=data_config.atomic_batch_n_variants,
            input_image_size=data_config.input_image_size,
            batch_size=data_config.train_batch_size,
            load_dof_angles=False,
            load_keypoint_positions=False,
            load_body_segment_maps=True,
            load_mesh_pose6d=True,
            shuffle=True,
            n_workers=data_config.n_workers,
            n_channels=3,
            pin_memory=True,
            drop_last=True,
            transform=self._compute_bodyseg_probs_transform,
        )

    def _init_validation_dataset_and_dataloader(
        self, data_config: config.TrainingDataConfig
    ):
        return synth_data.init_atomic_dataset_and_dataloader(
            data_dirs=data_config.val_data_dirs,
            atomic_batch_n_samples=data_config.atomic_batch_n_samples,
            atomic_batch_n_variants=data_config.atomic_batch_n_variants,
            input_image_size=data_config.input_image_size,
            batch_size=data_config.val_batch_size,
            load_dof_angles=False,
            load_keypoint_positions=False,
            load_body_segment_maps=True,
            load_mesh_pose6d=True,
            shuffle=False,
            n_workers=data_config.n_workers,
            n_channels=3,
            pin_memory=True,
            drop_last=True,
            transform=self._compute_bodyseg_probs_transform,
        )

    def _create_optimizer(self, optimizer_config: config.OptimizerConfig):
        params = [
            {
                "params": self.model.feature_extractor.parameters(),
                "lr": optimizer_config.learning_rate_encoder,
            },
            {
                "params": list(
                    chain(
                        # no dec_layer1 because we don't upsample all the way to 128
                        self.model.dec_layer2.parameters(),
                        self.model.dec_layer3.parameters(),
                        self.model.dec_layer4.parameters(),
                    )
                ),
                "lr": optimizer_config.learning_rate_deconv,
            },
            {
                "params": self.model.pose6d_heads.parameters(),
                "lr": optimizer_config.learning_rate_pose6d_heads,
            },
        ]
        if self.model.n_attention_gated_feature_channels > 0:
            params.append(
                {
                    "params": self.model.attention_heads.parameters(),
                    "lr": optimizer_config.learning_rate_attention_heads,
                }
            )

        optimizer = torch.optim.AdamW(
            params, weight_decay=optimizer_config.weight_decay
        )

        # Check if all parameters are covered
        n_params_optimizer = util.count_optimizer_parameters(optimizer)
        n_params_model = util.count_module_parameters(self.model)
        assert n_params_optimizer == n_params_model, (
            f"Number of parameters in optimizer ({n_params_optimizer}) does not match "
            f"number of parameters in model ({n_params_model})."
        )

        return optimizer

    def _check_amp_status_for_model_params(
        self, grad_scaler: torch.amp.GradScaler, subtitle: str = "Model parameters"
    ):
        return util.check_mixed_precision_status(
            self.use_float16,
            self.device,
            print_results=True,
            tensors={
                "feature_extractor_params": self.model.feature_extractor.parameters(),
                "decoder_params": chain(
                    self.model.dec_layer2.parameters(),
                    self.model.dec_layer3.parameters(),
                    self.model.dec_layer4.parameters(),
                ),
                "pose6d_heads_params": self.model.pose6d_heads.parameters(),
            },
            grad_scaler=grad_scaler,
            subtitle=subtitle,
        )

    def _check_amp_status_during_training(
        self,
        input_images: torch.Tensor,
        pred_pos: torch.Tensor,
        pred_quat: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        grad_scaler: torch.amp.GradScaler,
        subtitle: str = "Variables during training",
    ):
        return util.check_mixed_precision_status(
            self.use_float16,
            self.device,
            print_results=True,
            tensors={
                "input_images": input_images,
                "target_pos": target_pos,
                "target_quat": target_quat,
                "pred_pos": pred_pos,
                "pred_quat": pred_quat,
            },
            grad_scaler=grad_scaler,
            subtitle=subtitle,
        )

    @staticmethod
    def _save_checkpoint(
        checkpoint_path_stem: Path,
        model: Pose6DModel,
        loss: Pose6DLoss | None = None,
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
        logger.info(log_str)
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
        logger.info(log_str)

    def _get_half_batch(self, frames_batch, sim_data_batch):
        """Return half of the batch to save memory (for debugging only)."""
        half_batch_size = frames_batch.shape[0] // 2
        frames_batch = frames_batch[:half_batch_size, ...]
        sim_data_batch = {
            k: v[:half_batch_size, ...] for k, v in sim_data_batch.items()
        }
        return frames_batch, sim_data_batch
