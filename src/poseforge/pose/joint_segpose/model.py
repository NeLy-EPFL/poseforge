import logging
from pathlib import Path

import torch
import torch.nn as nn

import poseforge.pose.joint_segpose.config as config
from poseforge.pose.bodyseg.model import BodySegmentationModel
from poseforge.pose.common import ResNetFeatureExtractor
from poseforge.pose.keypoints3d.model import Pose2p5DModel


class JointSegPoseModel(nn.Module):
    """Joint 2.5D pose + body-segmentation model.

    A single ResNet18 ``ResNetFeatureExtractor`` instance is shared between
    ``Pose2p5DModel`` (self.pose) and ``BodySegmentationModel`` (self.bodyseg).
    Each task model still calls the backbone independently inside its own
    forward (two backbone passes per step), but gradients accumulate naturally
    through the shared parameters.

    Checkpoints are saved as a single payload ``{"pose": pose.state_dict(),
    "bodyseg": bodyseg.state_dict()}``; each sub-dict carries its own copy of
    the (shared) feature-extractor weights so the existing single-task
    inference scripts can load it unchanged via their patched loaders.
    """

    def __init__(
        self,
        pose_model: Pose2p5DModel,
        bodyseg_model: BodySegmentationModel,
    ):
        super().__init__()
        if pose_model.feature_extractor is not bodyseg_model.feature_extractor:
            raise ValueError(
                "JointSegPoseModel expects the two task models to share the same "
                "ResNetFeatureExtractor instance."
            )
        self.pose = pose_model
        self.bodyseg = bodyseg_model

    @property
    def feature_extractor(self) -> ResNetFeatureExtractor:
        return self.pose.feature_extractor

    @classmethod
    def create_architecture_from_config(
        cls, architecture_config: config.JointModelArchitectureConfig | Path | str
    ) -> "JointSegPoseModel":
        if isinstance(architecture_config, (Path, str)):
            architecture_config = config.JointModelArchitectureConfig.load(
                architecture_config
            )
            logging.info(
                f"Loaded joint architecture config from {architecture_config}"
            )

        shared_feature_extractor = ResNetFeatureExtractor()

        pose_cfg = architecture_config.pose
        pose_model = Pose2p5DModel(
            n_keypoints=pose_cfg.n_keypoints,
            feature_extractor=shared_feature_extractor,
            depth_n_bins=pose_cfg.depth_n_bins,
            depth_min=pose_cfg.depth_min,
            depth_max=pose_cfg.depth_max,
            xy_temperature=pose_cfg.xy_temperature,
            depth_temperature=pose_cfg.depth_temperature,
            upsample_core_out_channels=pose_cfg.upsample_core_out_channels,
            depth_hidden_channels=pose_cfg.depth_hidden_channels,
            confidence_method=pose_cfg.confidence_method,
            groupnorm_n_groups=pose_cfg.groupnorm_n_groups,
            pose_head_init_std=pose_cfg.pose_head_init_std,
            activation_noise_std=pose_cfg.activation_noise_std,
            decoder_spatial_dropout_p=pose_cfg.decoder_spatial_dropout_p,
            heatmap_n_hidden_layers=pose_cfg.heatmap_n_hidden_layers,
            heatmap_hidden_channels=pose_cfg.heatmap_hidden_channels,
        )

        seg_cfg = architecture_config.bodyseg
        bodyseg_model = BodySegmentationModel(
            n_classes=seg_cfg.n_classes,
            feature_extractor=shared_feature_extractor,
            final_upsampler_n_hidden_channels=seg_cfg.final_upsampler_n_hidden_channels,
            confidence_method=seg_cfg.confidence_method,
            activation_noise_std=seg_cfg.activation_noise_std,
        )

        return cls(pose_model=pose_model, bodyseg_model=bodyseg_model)

    def load_weights_from_config(
        self, weights_config: config.JointModelWeightsConfig | Path | str
    ) -> None:
        if isinstance(weights_config, (Path, str)):
            weights_config = config.JointModelWeightsConfig.load(weights_config)
            logging.info(f"Loaded joint weights config from {weights_config}")

        if (
            weights_config.feature_extractor_weights is None
            and weights_config.model_weights is None
        ):
            logging.warning(
                "JointModelWeightsConfig contains nothing useful. No action taken."
            )
            return

        # If a full joint-model checkpoint is given, load it directly.
        if weights_config.model_weights is not None:
            checkpoint_path = Path(weights_config.model_weights)
            if not checkpoint_path.is_file():
                raise ValueError(
                    f"Joint model weights path {checkpoint_path} is not a file"
                )
            payload = torch.load(checkpoint_path, map_location="cpu")
            if not (
                isinstance(payload, dict) and {"pose", "bodyseg"} <= set(payload.keys())
            ):
                raise ValueError(
                    f"Expected a joint checkpoint with 'pose' and 'bodyseg' keys at "
                    f"{checkpoint_path}, got keys: "
                    f"{list(payload.keys()) if isinstance(payload, dict) else type(payload)}"
                )
            self.pose.load_state_dict(payload["pose"])
            self.bodyseg.load_state_dict(payload["bodyseg"])
            logging.info("Loaded JointSegPoseModel weights from joint checkpoint")
            return

        # Otherwise, load the shared feature extractor only. Both sub-models
        # already point at the same instance, so a single in-place re-init
        # via Pose2p5DModel's loader updates both.
        shared_feature_extractor = ResNetFeatureExtractor(
            weights=weights_config.feature_extractor_weights
        )
        self.pose.feature_extractor = shared_feature_extractor
        self.bodyseg.feature_extractor = shared_feature_extractor
        logging.info(
            "Set up shared feature extractor from joint weights config "
            f"(weights={weights_config.feature_extractor_weights})"
        )

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "pose": self.pose(x),
            "bodyseg": self.bodyseg(x),
        }

    def save_joint_checkpoint(self, path: Path | str) -> None:
        """Write the joint payload ``{"pose": ..., "bodyseg": ...}`` to ``path``.

        Each sub-dict carries the (shared) feature-extractor weights, so the
        existing single-task ``load_weights_from_config`` paths slice their
        half out transparently.
        """
        torch.save(
            {
                "pose": self.pose.state_dict(),
                "bodyseg": self.bodyseg.state_dict(),
            },
            path,
        )
