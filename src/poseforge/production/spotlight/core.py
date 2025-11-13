import numpy as np
import pandas as pd
import torch
from pathlib import Path
from loguru import logger

import poseforge.production.spotlight.filtering as filtering
import poseforge.production.spotlight.keypoints3d as keypoints3d
import poseforge.production.spotlight.bodyseg as bodyseg
from poseforge.production.spotlight.paths import SpotlightRecordingPaths
from poseforge.production.config import load_config


class SpotlightRecordingProcessor:
    def __init__(
        self,
        trial_dir: Path | str,
        model_config_path: Path | str,
        with_muscle: bool = False,
        device: torch.device | str = None,
    ):
        self.trial_dir = trial_dir
        self.model_config_path = model_config_path
        self.with_muscle = with_muscle
        self.device = self._set_up_device(device)
        self.model_config = load_config(model_config_path)

        # Auto-derive paths to various files and validate their existence
        self.paths = SpotlightRecordingPaths(
            trial_basedir=trial_dir, with_muscle=with_muscle
        )

        # Record which steps have been completed
        self.usable_frames_detected = False
        self.keypoints3d_predicted = False
        self.inverse_kinematics_solved = False
        self.keypoints3d_ik_visualized = False
        self.bodyseg_predicted = False
        self.bodyseg_visualized = False

    @staticmethod
    def _set_up_device(device: torch.device | str = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        if not device.type == "cuda":
            logger.error("CUDA not available. CPU inference not supported (too slow).")
            raise RuntimeError("CUDA not available.")
        return device

    def detect_usable_frames(
        self,
        edge_tolerance_mm: float = 5.0,
        loading_batch_size: int = 128,
        loading_n_workers: int = 8,
        loading_buffer_size: int = 128,
        loading_cache_video_metadata: bool = True,
    ) -> pd.DataFrame:
        df = filtering.detect_usable_frames(
            usable_frames_output_path=self.paths.usable_frames,
            aligned_behavior_video_path=self.paths.aligned_behavior_video,
            alignment_metadata_path=self.paths.alignment_metadata,
            behavior_frames_metadata_path=self.paths.behavior_frames_metadata,
            behavior_cam_calib_path=self.paths.behavior_cam_calib,
            recorder_config_path=self.paths.recorder_config,
            flip_detection_model_config=self.model_config["flip_detection"],
            device=self.device,
            edge_tolerance_mm=edge_tolerance_mm,
            loading_batch_size=loading_batch_size,
            loading_n_workers=loading_n_workers,
            loading_buffer_size=loading_buffer_size,
            loading_cache_video_metadata=loading_cache_video_metadata,
        )
        self.usable_frames_detected = True
        return df

    def predict_keypoints3d(
        self,
        camera_pos: tuple[float, float, float] = (0.0, 0.0, -100.0),
        camera_fov_deg: float = 5.0,
        camera_rendering_size: tuple[int, int] = (464, 464),
        camera_rotation_euler: tuple[float, float, float] = (0, np.pi, -np.pi / 2),
        loading_batch_size: int = 128,
        loading_n_workers: int = 8,
        loading_buffer_size: int = 128,
        loading_cache_video_metadata: bool = True,
    ) -> None:
        keypoints3d.predict_keypoints3d(
            keypoints3d_output_path=self.paths.keypoints3d_prediction,
            aligned_behavior_video_path=self.paths.aligned_behavior_video,
            keypoints3d_model_config=self.model_config["keypoints3d"],
            device=self.device,
            camera_pos=camera_pos,
            camera_fov_deg=camera_fov_deg,
            camera_rendering_size=camera_rendering_size,
            camera_rotation_euler=camera_rotation_euler,
            loading_batch_size=loading_batch_size,
            loading_n_workers=loading_n_workers,
            loading_buffer_size=loading_buffer_size,
            loading_cache_video_metadata=loading_cache_video_metadata,
        )
        self.keypoints3d_predicted = True

    def solve_inverse_kinematics(self) -> None:
        if not self.keypoints3d_predicted:
            raise RuntimeError(
                "Keypoints3D must be predicted before solving inverse kinematics. "
                "Call .predict_keypoints3d() first."
            )
        pass  # TODO
        self.inverse_kinematics_solved = True

    def visualize_keypoints3d_ik(self) -> None:
        if not self.inverse_kinematics_solved:
            raise RuntimeError(
                "Inverse kinematics must be solved before visualizing. "
                "Call .solve_inverse_kinematics() first (which in turn requires "
                ".predict_keypoints3d() to be called first)."
            )
        pass  # TODO
        self.keypoints3d_ik_visualized = True

    def predict_body_segmentation(
        self,
        loading_batch_size: int = 128,
        loading_n_workers: int = 8,
        loading_buffer_size: int = 128,
        loading_cache_video_metadata: bool = True,
    ) -> None:
        bodyseg.predict_body_segmentation(
            bodyseg_output_path=self.paths.bodyseg_prediction,
            aligned_behavior_video_path=self.paths.aligned_behavior_video,
            bodyseg_model_config=self.model_config["bodyseg"],
            device=self.device,
            loading_batch_size=loading_batch_size,
            loading_n_workers=loading_n_workers,
            loading_buffer_size=loading_buffer_size,
            loading_cache_video_metadata=loading_cache_video_metadata,
        )
        self.bodyseg_predicted = True

    def visualize_bodyseg_predictions(self):
        if not self.bodyseg_predicted:
            raise RuntimeError(
                "Body segmentation maps must be predicted before visualizing. "
                "Call .predict_body_segmentation() first."
            )
        pass  # TODO
        self.bodyseg_visualized = True


if __name__ == "__main__":
    from loguru import logger
    from poseforge.util.sys import set_loguru_level

    set_loguru_level("INFO")

    model_config_path = Path(
        "~/projects/poseforge/src/poseforge/production/spotlight/config.yaml"
    ).expanduser()
    spotlight_recording_dir = Path("~/data/spotlight/20250613-fly1b-002").expanduser()
    recording = SpotlightRecordingProcessor(
        spotlight_recording_dir, model_config_path, with_muscle=True
    )
    recording.detect_usable_frames(edge_tolerance_mm=5.0, loading_n_workers=8)
    recording.predict_keypoints3d(loading_n_workers=8)
    recording.solve_inverse_kinematics()
    recording.visualize_keypoints3d_ik()
    recording.predict_body_segmentation(loading_n_workers=8)
    recording.visualize_bodyseg_predictions()
