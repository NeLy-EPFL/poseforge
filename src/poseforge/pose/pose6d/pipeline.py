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


import poseforge.pose.pose6d.config as config
from poseforge.pose.pose6d.model import Pose6DModel, Pose6DLoss
from poseforge.pose.data.synthetic import (
    init_atomic_dataset_and_dataloader,
    atomic_batches_to_simple_batch,
)
from poseforge.util import (
    set_random_seed,
    check_mixed_precision_status,
    count_optimizer_parameters,
    count_module_parameters,
    clear_memory_cache,
)


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

    def train(
        self,
        n_epochs: int,
        data_config: config.TrainingDataConfig,
        optimizer_config: config.OptimizerConfig,
        artifacts_config: config.TrainingArtifactsConfig,
        seed: int = 42,
    ):
        # Set random seed for reproducibility
        set_random_seed(seed)

        # Set up training and validation data
        train_ds, train_loader = self._init_training_dataset_and_dataloader(data_config)
        val_ds, val_loader = self._init_validation_dataset_and_dataloader(data_config)

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
                frames, sim_data = atomic_batches_to_simple_batch(
                    atomic_batches_frames, atomic_batches_sim_data, device=self.device
                )
                ...

    def _init_training_dataset_and_dataloader(
        self, data_config: config.TrainingDataConfig
    ):
        return init_atomic_dataset_and_dataloader(
            data_dirs=data_config.train_data_dirs,
            atomic_batch_n_samples=data_config.atomic_batch_n_samples,
            atomic_batch_n_variants=data_config.atomic_batch_n_variants,
            input_image_size=data_config.input_image_size,
            batch_size=data_config.train_batch_size,
            load_dof_angles=False,
            load_keypoint_positions=False,
            load_body_segment_maps=False,
            shuffle=True,
            n_workers=data_config.n_workers,
            n_channels=3,
            pin_memory=True,
            drop_last=True,
        )

    def _init_validation_dataset_and_dataloader(
        self, data_config: config.TrainingDataConfig
    ):
        return init_atomic_dataset_and_dataloader(
            data_dirs=data_config.val_data_dirs,
            atomic_batch_n_samples=data_config.atomic_batch_n_samples,
            atomic_batch_n_variants=data_config.atomic_batch_n_variants,
            input_image_size=data_config.input_image_size,
            batch_size=data_config.val_batch_size,
            load_dof_angles=False,
            load_keypoint_positions=False,
            load_body_segment_maps=False,
            shuffle=False,
            n_workers=data_config.n_workers,
            n_channels=3,
            pin_memory=True,
            drop_last=True,
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
                        self.model.dec_layer1.parameters(),
                        self.model.dec_layer2.parameters(),
                        self.model.dec_layer3.parameters(),
                        self.model.dec_layer4.parameters(),
                    )
                ),
                "lr": optimizer_config.learning_rate_deconv,
            },
            {
                "params": list(
                    chain(
                        self.model.pose6d_heads.parameters(),
                    )
                ),
                "lr": optimizer_config.learning_rate_pose6d_heads,
            },
        ]

        optimizer = torch.optim.AdamW(
            params, weight_decay=optimizer_config.weight_decay
        )

        # Check if all parameters are covered
        n_params_optimizer = count_optimizer_parameters(optimizer)
        n_params_model = count_module_parameters(self.model)
        assert n_params_optimizer == n_params_model, (
            f"Number of parameters in optimizer ({n_params_optimizer}) does not match "
            f"number of parameters in model ({n_params_model})."
        )

        return optimizer

    def _check_amp_status_for_model_params(
        self, grad_scaler: torch.amp.GradScaler, subtitle: str = "Model parameters"
    ):
        return check_mixed_precision_status(
            self.use_float16,
            self.device,
            print_results=True,
            tensors={
                "feature_extractor_params": self.model.feature_extractor.parameters(),
                "decoder_params": chain(
                    self.model.dec_layer1.parameters(),
                    self.model.dec_layer2.parameters(),
                    self.model.dec_layer3.parameters(),
                    self.model.dec_layer4.parameters(),
                ),
                "pose6d_heads_params": self.model.pose6d_heads.parameters(),
            },
            grad_scaler=grad_scaler,
            subtitle=subtitle,
        )

    # def _check_amp_status_during_training(
    #     self,
    #     input_images: torch.Tensor,
    #     target: torch.Tensor,
    #     pred_dict: torch.Tensor,
    #     grad_scaler: torch.amp.GradScaler,
    #     subtitle: str = "Variables during training",
    # ):
    #     return check_mixed_precision_status(
    #         self.use_float16,
    #         self.device,
    #         print_results=True,
    #         tensors={
    #             "input_images": input_images,
    #             "target": target,
    #             "pred": pred_dict["logits"],
    #             "pred_conf": pred_dict["confidence"],
    #         },
    #         grad_scaler=grad_scaler,
    #         subtitle=subtitle,
    #     )
