import logging

logging_level = logging.INFO
logging.basicConfig(
    level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s"
)

import torch
from pathlib import Path
from torchsummary import summary

import poseforge.pose.simsiam.config as config
from poseforge.pose.simsiam import (
    SimSiamPretrainingPipeline,
    SimSiamPretrainingModel,
    SimSiamLoss,
)
from poseforge.util import set_random_seed, get_hardware_availability


def pretrain_simsiam_model(
    n_epochs: int,
    simsiam_architecture_config: config.SimSiamArchitectureConfig,
    model_weights_config: config.ModelWeightsConfig,
    training_data_config: config.TrainingDataConfig,
    optimizer_config: config.OptimizerConfig,
    training_artifacts_config: config.TrainingArtifactsConfig,
    seed: int = 42,
) -> None:
    """Pretrain a ResNet feature extractor with the SimSiam objective.
    Uses the same atomic-batch dataset, the same aligned-crop logic, the
    same alignment metric, and the same checkpoint format as the InfoNCE
    pipeline so the two methods can be compared head-to-head on identical
    infrastructure."""
    set_random_seed(seed)
    hardware_avail = get_hardware_availability(check_gpu=True, print_results=True)
    if len(hardware_avail["gpus"]) == 0:
        raise RuntimeError("No GPU available for training")
    torch.backends.cudnn.benchmark = True

    configs_dir = Path(training_artifacts_config.output_basedir) / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    simsiam_architecture_config.save(configs_dir / "model_architecture_config.yaml")
    if model_weights_config is not None:
        model_weights_config.save(configs_dir / "model_weights_config.yaml")
    training_data_config.save(configs_dir / "data_config.yaml")
    optimizer_config.save(configs_dir / "optimizer_config.yaml")
    training_artifacts_config.save(configs_dir / "artifacts_config.yaml")

    model = SimSiamPretrainingModel.create_architecture_from_config(
        architecture_config=simsiam_architecture_config
    )
    model.load_weights_from_config(weights_config=model_weights_config)

    print("========== Feature Extractor Summary ==========")
    summary_input_size = (
        tuple(training_data_config.crop_size)
        if training_data_config.crop_size is not None
        else tuple(training_data_config.image_size)
    )
    summary(model.feature_extractor.cuda(), (3, *summary_input_size))
    print("=========== Projection Head Summary ===========")
    summary(
        model.projection_head.cuda(),
        (model.feature_extractor.output_channels,),
    )
    print("=============== Predictor Summary ==============")
    summary(
        model.predictor.cuda(),
        (simsiam_architecture_config.projection_head_output_dim,),
    )

    pipeline = SimSiamPretrainingPipeline(
        simsiam_model=model,
        simsiam_loss_func=SimSiamLoss.create(),
        device="cuda",
        use_float16=True,
    )

    pipeline.train(
        n_epochs=n_epochs,
        data_config=training_data_config,
        optimizer_config=optimizer_config,
        artifacts_config=training_artifacts_config,
        seed=seed,
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(
        pretrain_simsiam_model,
        prog=f"python {Path(__file__).name}",
        description=(
            "Pretrain a ResNet feature extractor using SimSiam on "
            "synthetic-style-variant atomic batches. Designed to be a drop-in "
            "head-to-head alternative to the InfoNCE pipeline."
        ),
    )
