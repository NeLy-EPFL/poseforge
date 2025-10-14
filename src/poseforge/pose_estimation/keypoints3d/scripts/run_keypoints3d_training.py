import logging

logging_level = logging.INFO
logging.basicConfig(
    level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s"
)

import torch
from pathlib import Path
from torchsummary import summary

import poseforge.pose_estimation.keypoints3d.config as config
from poseforge.pose_estimation.keypoints3d import (
    Pose2p5DModel,
    Pose2p5DLoss,
    Pose2p5DPipeline,
)
from poseforge.util import get_hardware_availability


def _setup_model(
    architecture_config: config.ModelArchitectureConfig,
    weights_config: config.ModelWeightsConfig | None,
) -> Pose2p5DModel:
    model = Pose2p5DModel.create_architecture_from_config(architecture_config)
    if weights_config is not None:
        model.load_weights_from_config(weights_config)
    logging.info("Set up model, incorporating feature extractor above")
    return model


def _setup_loss_func(loss_config: config.LossConfig) -> Pose2p5DLoss:
    loss_func = Pose2p5DLoss.create_from_config(loss_config)
    logging.info("Set up 2.5D pose estimation loss function")
    return loss_func


def train_keypoints3d_model(
    n_epochs: int,
    model_architecture_config: config.ModelArchitectureConfig,
    model_weights_config: config.ModelWeightsConfig,
    loss_config: config.LossConfig,
    training_data_config: config.TrainingDataConfig,
    optimizer_config: config.OptimizerConfig,
    training_artifacts_config: config.TrainingArtifactsConfig,
    seed: int = 42,
) -> None:
    """Train a 3D keypoint detection model, typically using a pretrained
    feature extractor.

    Args:
        n_epochs: Number of epochs to train for.
        model_architecture_config: Configuration for model architecture.
        model_weights_config: Configuration for model weights to load.
            If None, will initialize model weights randomly.
        loss_config: Configuration for loss function.
        training_data_config: Configuration for training and validation
            data.
        optimizer_config: Configuration for optimizer and learning rate
            schedule.
        training_artifacts_config: Configuration for saving training
            artifacts.
        seed: Random seed for reproducibility.
    """
    # System setup
    hardware_avail = get_hardware_availability(check_gpu=True, print_results=True)
    if len(hardware_avail["gpus"]) == 0:
        raise RuntimeError("No GPU available for training")
    torch.backends.cudnn.benchmark = True

    # Save configs
    configs_dir = Path(training_artifacts_config.output_basedir) / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    model_architecture_config.save(configs_dir / "model_architecture_config.yaml")
    if model_weights_config is not None:
        model_weights_config.save(configs_dir / "model_weights_config.yaml")
    loss_config.save(configs_dir / "loss_config.yaml")
    training_data_config.save(configs_dir / "data_config.yaml")
    optimizer_config.save(configs_dir / "optimizer_config.yaml")
    training_artifacts_config.save(configs_dir / "artifacts_config.yaml")

    # Initialize model and loss function
    pose_model = _setup_model(model_architecture_config, model_weights_config)
    criterion = _setup_loss_func(loss_config)
    print("========== Model Summary ==========")
    summary(
        pose_model, input_size=(3, *training_data_config.input_image_size), device="cpu"
    )

    # Initialize learning pipeline
    pipeline = Pose2p5DPipeline(
        model=pose_model, loss_func=criterion, device="cuda", use_float16=True
    )

    # Train the model
    pipeline.train(
        n_epochs=n_epochs,
        data_config=training_data_config,
        optimizer_config=optimizer_config,
        artifacts_config=training_artifacts_config,
        seed=seed,
    )

    logging.info("Training complete")


if __name__ == "__main__":
    import tyro

    tyro.cli(
        train_keypoints3d_model,
        prog=f"python {Path(__file__).name}",
        description="Train a 3D keypoint detection model using pretrained feature extractor.",
    )

    # model_architecture_config = config.ModelArchitectureConfig()
    # checkpoint_path = "bulk_data/pose_estimation/contrastive_pretraining/trial_20251011a/checkpoints/checkpoint_epoch009_step003055.feature_extractor.pth"
    # model_weights_config = config.ModelWeightsConfig(
    #     feature_extractor_weights=str(checkpoint_path),
    #     model_weights=None,
    # )
    # loss_config = config.LossConfig()
    # data_basedir = Path("bulk_data/pose_estimation/atomic_batches")
    # train_data_dirs = [
    #     data_basedir / f"BO_Gal4_fly{fly}_trial{trial:03d}"
    #     for fly in range(1, 5)  # flies 1-4
    #     for trial in range(1, 6)  # trials 1-5
    # ]
    # val_data_dirs = [data_basedir / f"BO_Gal4_fly1_trial001"]
    # training_data_config = config.TrainingDataConfig(
    #     train_data_dirs=[str(path) for path in train_data_dirs],
    #     val_data_dirs=[str(path) for path in val_data_dirs],
    #     input_image_size=(256, 256),
    #     atomic_batch_n_samples=32,
    #     atomic_batch_n_variants=4,
    #     train_batch_size=32,
    #     val_batch_size=32,
    #     num_workers=8,
    # )
    # optimizer_config = config.OptimizerConfig()
    # training_artifacts_config = config.TrainingArtifactsConfig(
    #     output_basedir="bulk_data/pose_estimation/keypoints3d/trial_20251013a/",
    #     logging_interval=100,
    #     checkpoint_interval=100,
    #     validation_interval=100,
    #     n_batches_per_validation=30,
    # )
    # train_keypoints3d_model(
    #     n_epochs=30,
    #     model_architecture_config=model_architecture_config,
    #     model_weights_config=model_weights_config,
    #     loss_config=loss_config,
    #     training_data_config=training_data_config,
    #     optimizer_config=optimizer_config,
    #     training_artifacts_config=training_artifacts_config,
    #     seed=42,
    # )
