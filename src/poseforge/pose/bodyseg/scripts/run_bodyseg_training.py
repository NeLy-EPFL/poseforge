import logging

logging_level = logging.INFO
logging.basicConfig(
    level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s"
)

import torch
from pathlib import Path
from torchsummary import summary

import poseforge.pose.bodyseg.config as config
from poseforge.pose.bodyseg import (
    BodySegmentationModel,
    CombinedDiceCELoss,
    BodySegmentationPipeline,
)
from poseforge.util import get_hardware_availability


def setup_model(
    architecture_config: config.ModelArchitectureConfig,
    weights_config: config.ModelWeightsConfig | None,
) -> BodySegmentationModel:
    model = BodySegmentationModel.create_architecture_from_config(architecture_config)
    logging.info(
        f"Model architecture set up based on architecture config {architecture_config}"
    )
    if weights_config is not None:
        model.load_weights_from_config(weights_config)
    logging.info(f"Loaded weights into model based on weights config {weights_config}")
    return model


def setup_loss_func(loss_config: config.LossConfig) -> CombinedDiceCELoss:
    loss_func = CombinedDiceCELoss.create_from_config(loss_config)
    logging.info("Set up combined loss function")
    return loss_func


def save_configs(
    configs_dir: Path,
    *,
    model_architecture_config: config.ModelArchitectureConfig,
    loss_config: config.LossConfig,
    training_data_config: config.TrainingDataConfig,
    optimizer_config: config.OptimizerConfig,
    training_artifacts_config: config.TrainingArtifactsConfig,
    model_weights_config: config.ModelWeightsConfig | None = None,
) -> None:
    configs_dir.mkdir(parents=True, exist_ok=True)
    model_architecture_config.save(configs_dir / "model_architecture_config.yaml")
    loss_config.save(configs_dir / "loss_config.yaml")
    training_data_config.save(configs_dir / "data_config.yaml")
    optimizer_config.save(configs_dir / "optimizer_config.yaml")
    training_artifacts_config.save(configs_dir / "artifacts_config.yaml")
    if model_weights_config is not None:
        model_weights_config.save(configs_dir / "model_weights_config.yaml")


def train_bodyseg_model(
    n_epochs: int,
    model_architecture_config: config.ModelArchitectureConfig,
    model_weights_config: config.ModelWeightsConfig,
    loss_config: config.LossConfig,
    training_data_config: config.TrainingDataConfig,
    optimizer_config: config.OptimizerConfig,
    training_artifacts_config: config.TrainingArtifactsConfig,
    seed: int = 42,
    half_batch_size_for_debugging: bool = False,
) -> None:
    # System setup
    hardware_avail = get_hardware_availability(check_gpu=True, print_results=True)
    if len(hardware_avail["gpus"]) == 0:
        raise RuntimeError("No GPU available for training")
    torch.backends.cudnn.benchmark = True

    # Save configs
    save_configs(
        Path(training_artifacts_config.output_basedir) / "configs",
        model_architecture_config=model_architecture_config,
        model_weights_config=model_weights_config,
        loss_config=loss_config,
        training_data_config=training_data_config,
        optimizer_config=optimizer_config,
        training_artifacts_config=training_artifacts_config,
    )

    # Initialize model and loss function
    model = setup_model(model_architecture_config, model_weights_config)
    criterion = setup_loss_func(loss_config)

    # Print model summary
    print_model_summary(training_data_config, model)

    # Initialize learning pipeline
    pipeline = BodySegmentationPipeline(
        model=model, loss_func=criterion, device="cuda", use_float16=True
    )

    # Train the model
    pipeline.train(
        n_epochs=n_epochs,
        data_config=training_data_config,
        optimizer_config=optimizer_config,
        artifacts_config=training_artifacts_config,
        seed=seed,
        half_batch_size_for_debugging=half_batch_size_for_debugging,
    )

    logging.info("Training complete")

def print_model_summary(training_data_config, model):
    down_in_dim = (3, *training_data_config.input_image_size)
    print("============== Full Model Summary ===============")
    summary(model, down_in_dim, device="cpu")
    print("=========== Feature Extractor Summary ===========")
    summary(model.feature_extractor, down_in_dim, device="cpu")


if __name__ == "__main__":
    import tyro

    tyro.cli(
        train_bodyseg_model,
        prog=f"python {Path(__file__).name}",
        description="Train a 3D body segmentation model using pretrained feature extractor.",
    )

    # Example using native Python function calls:
    # model_architecture_config = config.ModelArchitectureConfig()
    # model_weights_config = config.ModelWeightsConfig(
    #     feature_extractor_weights="bulk_data/pose_estimation/contrastive_pretraining/trial_20251011a/checkpoints/checkpoint_epoch001_step002000.feature_extractor.pth",
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
    #     n_workers=8,
    # )
    # optimizer_config = config.OptimizerConfig()
    # training_artifacts_config = config.TrainingArtifactsConfig(
    #     output_basedir="bulk_data/pose_estimation/bodyseg/trial_20251011b/",
    #     logging_interval=10,  # 1000
    #     checkpoint_interval=100,  # 1000
    #     validation_interval=100,  # 1000
    #     n_batches_per_validation=100,  # 300
    # )
    # train_bodyseg_model(
    #     n_epochs=10,
    #     model_architecture_config=model_architecture_config,
    #     model_weights_config=model_weights_config,
    #     loss_config=loss_config,
    #     training_data_config=training_data_config,
    #     optimizer_config=optimizer_config,
    #     training_artifacts_config=training_artifacts_config,
    #     seed=42,
    #     half_batch_size_for_debugging=True,
    # )
