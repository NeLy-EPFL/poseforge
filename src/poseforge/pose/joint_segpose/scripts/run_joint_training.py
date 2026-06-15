import logging

logging_level = logging.INFO
logging.basicConfig(
    level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s"
)

from pathlib import Path

import torch
from torchsummary import summary

import poseforge.pose.joint_segpose.config as config
import poseforge.pose.keypoints3d.config as keypoints3d_config
from poseforge.pose.bodyseg.model import CombinedDiceCELoss
from poseforge.pose.bodyseg.pipeline import BodySegmentationPipeline
from poseforge.pose.joint_segpose import JointSegPoseModel, JointSegPosePipeline
from poseforge.pose.keypoints3d.model import Pose2p5DLoss
from poseforge.util import get_hardware_availability


def train_joint_segpose_model(
    n_epochs: int,
    model_architecture_config: config.JointModelArchitectureConfig,
    model_weights_config: config.JointModelWeightsConfig,
    loss_config: config.JointLossConfig,
    training_data_config: config.JointTrainingDataConfig,
    optimizer_config: config.JointOptimizerConfig,
    training_artifacts_config: keypoints3d_config.TrainingArtifactsConfig,
    selected_original_class_indices: list[int],
    seed: int = 42,
) -> None:
    """Jointly train pose (2.5D keypoints) and body-part segmentation on top
    of a single shared ResNet18 backbone.

    A single canonical checkpoint per step is written; the existing pose
    inference script (`run_image_inference.py`) and body-seg inference script
    (`run_bodyseg_inference.py`) both load it unchanged by slicing their half
    out of the joint payload.
    """
    hardware_avail = get_hardware_availability(check_gpu=True, print_results=True)
    if len(hardware_avail["gpus"]) == 0:
        raise RuntimeError("No GPU available for training")
    torch.backends.cudnn.benchmark = True

    configs_dir = Path(training_artifacts_config.output_basedir) / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    model_architecture_config.save(configs_dir / "model_architecture_config.yaml")
    model_weights_config.save(configs_dir / "model_weights_config.yaml")
    loss_config.save(configs_dir / "loss_config.yaml")
    training_data_config.save(configs_dir / "data_config.yaml")
    optimizer_config.save(configs_dir / "optimizer_config.yaml")
    training_artifacts_config.save(configs_dir / "artifacts_config.yaml")

    if (
        len(selected_original_class_indices) + 1
        != model_architecture_config.bodyseg.n_classes
    ):
        raise ValueError(
            "selected_original_class_indices must contain exactly bodyseg n_classes - 1 "
            f"ids (got {len(selected_original_class_indices)} ids for n_classes="
            f"{model_architecture_config.bodyseg.n_classes})"
        )

    model = JointSegPoseModel.create_architecture_from_config(model_architecture_config)
    model.load_weights_from_config(model_weights_config)
    logging.info("Set up JointSegPoseModel (shared backbone + pose head + bodyseg head)")

    # torchsummary cannot introspect modules that return dicts, so we only
    # summarize the shared feature extractor (which returns a tensor). The
    # downstream pose and bodyseg heads each return a dict — see their
    # individual training scripts if you want to inspect them in isolation.
    print("=========== Shared Feature Extractor Summary ===========")
    summary(
        model.feature_extractor,
        input_size=(3, *training_data_config.input_image_size),
        device="cpu",
    )

    pose_loss = Pose2p5DLoss.create_from_config(loss_config.pose)
    bodyseg_loss = CombinedDiceCELoss.create_from_config(loss_config.bodyseg)
    logging.info("Set up joint loss (Pose2p5DLoss + CombinedDiceCELoss)")

    target_label_mapper = BodySegmentationPipeline.create_target_label_mapper(
        selected_original_class_indices=selected_original_class_indices,
        n_output_classes=model_architecture_config.bodyseg.n_classes,
    )

    pipeline = JointSegPosePipeline(
        model=model,
        pose_loss=pose_loss,
        bodyseg_loss=bodyseg_loss,
        weight_pose=loss_config.weight_pose,
        weight_bodyseg=loss_config.weight_bodyseg,
        device="cuda",
        use_float16=True,
        target_label_mapper=target_label_mapper,
    )

    pipeline.train(
        n_epochs=n_epochs,
        data_config=training_data_config,
        optimizer_config=optimizer_config,
        artifacts_config=training_artifacts_config,
        seed=seed,
    )

    logging.info("Joint training complete")


if __name__ == "__main__":
    import tyro

    tyro.cli(
        train_joint_segpose_model,
        prog=f"python {Path(__file__).name}",
        description=(
            "Jointly train a 2.5D keypoint head and a body-part segmentation head on "
            "top of a shared (contrastively pretrained) ResNet18 backbone."
        ),
    )
