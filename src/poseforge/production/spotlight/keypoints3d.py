import torch
import numpy as np
import torchvision.transforms as transforms
import h5py
from time import perf_counter
from tqdm import tqdm
from pathlib import Path
from typing import Any
from loguru import logger

from pvio.torch_tools import SimpleVideoCollectionLoader

import poseforge.pose.keypoints3d as keypoints3d
from poseforge.pose.camera import CameraToWorldMapper
from poseforge.neuromechfly.constants import keypoint_segments_canonical


def predict_keypoints3d(
    *,
    keypoints3d_output_path: Path,
    aligned_behavior_video_path: Path,
    keypoints3d_model_config: dict[str, Any],
    device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
    camera_pos: tuple[float, float, float] = (0.0, 0.0, -100.0),
    camera_fov_deg: float = 5.0,
    camera_rendering_size: tuple[int, int] = (464, 464),
    camera_rotation_euler: tuple[float, float, float] = (0, np.pi, -np.pi / 2),
    loading_batch_size: int = 128,
    loading_n_workers: int = 8,
    loading_buffer_size: int = 128,
    loading_cache_video_metadata: bool = True,
) -> None:
    logger.info("Estimating 3D keypoint positions for behavior video")

    # Set up model and pipeline
    architecture_config_path = Path(
        keypoints3d_model_config["architecture_config"]
    ).expanduser()
    logger.info(
        f"Setting up 3D keypoints model from architecture config "
        f"{architecture_config_path}"
    )
    model = keypoints3d.Pose2p5DModel.create_architecture_from_config(
        architecture_config_path
    ).to(device)
    ckpt_path = Path(keypoints3d_model_config["checkpoint"]).expanduser()
    logger.info(f"Loading 3D keypoints model weights from {ckpt_path}")
    weights_config = keypoints3d.config.ModelWeightsConfig(model_weights=ckpt_path)
    model.load_weights_from_config(weights_config)
    logger.info("Creating 3D keypoints inference pipeline")
    pipeline = keypoints3d.Pose2p5DPipeline(model, device=device, use_float16=True)

    # Set up camera mapper
    cam_mapper = CameraToWorldMapper(
        camera_pos, camera_fov_deg, camera_rendering_size, camera_rotation_euler
    )

    # Create video loader
    logger.info("Creating video loader for 3D keypoints prediction")
    working_size = keypoints3d_model_config["working_size"]
    video_loader = SimpleVideoCollectionLoader(
        [aligned_behavior_video_path],
        transform=transforms.Resize((working_size, working_size)),
        batch_size=loading_batch_size,
        num_workers=loading_n_workers,
        buffer_size=loading_buffer_size,
        use_cached_video_metadata=loading_cache_video_metadata,
    )

    # Create output file
    logger.info(
        f"Creating H5 file for 3D keypoint position predictions: "
        f"{keypoints3d_output_path}"
    )
    n_frames_total = len(video_loader.dataset)
    with h5py.File(keypoints3d_output_path, "w") as f:
        keypoint_names = keypoints3d_model_config["keypoint_names"]
        assert keypoint_names == keypoint_segments_canonical, (
            "Keypoint names in model config do not match expected canonical "
            "keypoint names from "
            "poseforge.neuromechfly.constants.keypoint_segments_canonical"
        )
        f.attrs["keypoint_names"] = keypoint_names
        n_keypoints = len(keypoint_names)

        def _create_dataset(name, shape):
            return f.create_dataset(
                name, shape=shape, dtype="float32", compression="gzip"
            )

        ds_pred_xy = _create_dataset("pred_xy", (n_frames_total, n_keypoints, 2))
        ds_pred_depth = _create_dataset("pred_depth", (n_frames_total, n_keypoints))
        ds_conf_xy = _create_dataset("conf_xy", (n_frames_total, n_keypoints))
        ds_conf_depth = _create_dataset("conf_depth", (n_frames_total, n_keypoints))
        ds_pred_world_xyz = _create_dataset(
            "pred_world_xyz", (n_frames_total, n_keypoints, 3)
        )

        # Run inference
        logger.info("Running inference for 3D keypoints prediction")
        for batch in tqdm(
            video_loader, desc="Predicting keypoints3d", unit="batch", disable=None
        ):
            # Forward pass
            # No need to move data to GPU and back, Pipeline.inference handles that
            frames = batch["frames"]
            frame_ids = batch["frame_indices"]
            assert (np.array(batch["video_indices"]) == 0).all()
            pred_dict = pipeline.inference(frames)
            batch_pred_xy = pred_dict["pred_xy"].numpy()  # (B, n_keypoints, 2)
            batch_pred_depth = pred_dict["pred_depth"].numpy()  # (B, n_keypoints)
            batch_conf_xy = pred_dict["conf_xy"].numpy()  # (B, n_keypoints)
            batch_conf_depth = pred_dict["conf_depth"].numpy()  # (B, n_keypoints)

            # So far, xy are pixel coords and depths are distances from camera in mm
            # By nature of alignment step in spotlight_tools.postprocessing, the
            # camera is "fixed" relative to the fly. Using the camera intrinsics and
            # extrinsics provided in the arguments, map predictions to xyz world
            # coords in mm. Shape should be (B, n_keypoints, 3)
            batch_pred_world_xyz = cam_mapper(batch_pred_xy, batch_pred_depth)

            # Save to H5 file
            start_time = perf_counter()
            ds_pred_xy[frame_ids, ...] = batch_pred_xy
            ds_pred_depth[frame_ids, ...] = batch_pred_depth
            ds_conf_xy[frame_ids, ...] = batch_conf_xy
            ds_conf_depth[frame_ids, ...] = batch_conf_depth
            ds_pred_world_xyz[frame_ids, ...] = batch_pred_world_xyz
            elapsed = perf_counter() - start_time
            logger.debug(f"Saved output for {len(frame_ids)} frames in {elapsed:.3f}s")
    
    logger.info("3D keypoints prediction complete")