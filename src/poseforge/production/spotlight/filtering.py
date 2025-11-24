import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import h5py
import yaml
from typing import Any
from tqdm import tqdm
from pathlib import Path
from loguru import logger

from spotlight_tools.calibration import SpotlightPositionMapper
from pvio.torch_tools import SimpleVideoCollectionLoader

import poseforge.spotlight.flip_detection as flip_detection


def detect_usable_frames(
    *,
    usable_frames_output_path: Path,
    aligned_behavior_video_path: Path,
    alignment_metadata_path: Path,
    behavior_frames_metadata_path: Path,
    behavior_cam_calib_path: Path,
    recorder_config_path: Path,
    flip_detection_model_config: dict[str, Any],
    device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
    edge_tolerance_mm: float = 5.0,
    loading_batch_size: int = 128,
    loading_n_workers: int = 8,
    loading_buffer_size: int = 128,
    loading_cache_video_metadata: bool = True,
) -> pd.DataFrame:
    logger.info("Detecting usable frames based on flip and edge criteria")
    df_flipped = _detect_flipped_frames(
        aligned_behavior_video_path=aligned_behavior_video_path,
        flip_detection_model_config=flip_detection_model_config,
        device=device,
        loading_batch_size=loading_batch_size,
        loading_n_workers=loading_n_workers,
        loading_buffer_size=loading_buffer_size,
        loading_cache_video_metadata=loading_cache_video_metadata,
    )
    df_edge = _detect_close_to_edge(
        alignment_metadata_path=alignment_metadata_path,
        behavior_frames_metadata_path=behavior_frames_metadata_path,
        behavior_cam_calib_path=behavior_cam_calib_path,
        recorder_config_path=recorder_config_path,
        edge_tolerance_mm=edge_tolerance_mm,
    )
    df = pd.merge(df_flipped, df_edge, on="behavior_frameid", how="inner")

    is_not_flipped = df["flip_detection"] == "not_flipped"
    is_not_too_close = ~df["too_close_to_edge"]
    df["usable"] = is_not_flipped & is_not_too_close
    df.to_csv(usable_frames_output_path, index=False)
    logger.info(
        f"Out of {len(df)} frames, {is_not_flipped.sum()} frames are not flipped, "
        f"{is_not_too_close.sum()} frames are not too close to edge, and "
        f"{df['usable'].sum()} frames meet both criteria. "
        f"Saved usable frames metadata to {usable_frames_output_path}"
    )
    return df


def _detect_flipped_frames(
    *,
    aligned_behavior_video_path: Path,
    flip_detection_model_config: dict[str, Any],
    device: torch.device | str,
    loading_batch_size: int,
    loading_n_workers: int,
    loading_buffer_size: int,
    loading_cache_video_metadata: bool,
) -> pd.DataFrame:
    # Set up flip detection model
    logger.info("Setting up flip detection model")
    model = flip_detection.FlipDetectionCNN(
        **flip_detection_model_config["init_params"]
    )
    checkpoint_path = Path(flip_detection_model_config["checkpoint"]).expanduser()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    class_labels = np.array(flip_detection_model_config["output_labels"])

    # Create video loader
    logger.info("Creating video loader for flip detection")
    working_size = flip_detection_model_config["working_size"]
    transform_ = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((working_size, working_size)),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    video_loader = SimpleVideoCollectionLoader(
        [aligned_behavior_video_path],
        transform=transform_,
        batch_size=loading_batch_size,
        num_workers=loading_n_workers,
        buffer_size=loading_buffer_size,
        use_cached_video_metadata=loading_cache_video_metadata,
    )

    # Run inference
    logger.info("Running inference for flip detection")
    results_all = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            video_loader, desc="Detecting flips", unit="batch", disable=None
        ):
            frames = batch["frames"]  # BCHW tensor
            frame_indices = batch["frame_indices"]
            logits = model(frames.to(device))
            batch_probs = torch.softmax(logits, dim=1)
            batch_confs, batch_preds = torch.max(batch_probs, dim=1)
            batch_confs = batch_confs.detach().cpu().numpy()
            batch_preds = batch_preds.detach().cpu().numpy()
            batch_pred_labels = class_labels[batch_preds]
            for i, frameid in enumerate(frame_indices):
                results_all.append([frameid, batch_pred_labels[i], batch_confs[i]])

    # Convert results to dataframe
    pred_df = pd.DataFrame(
        results_all, columns=["behavior_frameid", "flip_detection", "flip_conf"]
    )
    pred_df["behavior_frameid"] = pred_df["behavior_frameid"].astype("uint32")
    pred_df["flip_detection"] = pred_df["flip_detection"].astype("category")
    pred_df["flip_conf"] = pred_df["flip_conf"].astype("float32")
    pred_df = pred_df.sort_values("behavior_frameid")
    assert np.all(
        pred_df["behavior_frameid"] == np.arange(len(pred_df))
    ), "Frame indices are not continuous"
    return pred_df


def _detect_close_to_edge(
    *,
    alignment_metadata_path: Path,
    behavior_frames_metadata_path: Path,
    behavior_cam_calib_path: Path,
    recorder_config_path: Path,
    edge_tolerance_mm: float,
) -> pd.DataFrame:
    # Load keypoint positions predicted by 2D pose model
    logger.info("Loading keypoint positions from 2D pose model")
    with h5py.File(alignment_metadata_path, "r") as f:
        ds = f["keypoints_xy_pre_alignment"]
        keypoints = ds.attrs["keypoint_names"]
        keypoints_exp = ["neck", "thorax", "abdomen tip"]
        assert keypoints.tolist() == keypoints_exp, "Unexpected keypoint names"
        all_keypoint_pos_pixel_xy = ds[:]  # (n_frames, num_keypoints=3, 2={x,y})

    # Load stage positions
    logger.info("Loading stage positions from behavior frames metadata")
    behavior_frames_metadata_df = pd.read_csv(behavior_frames_metadata_path)
    stage_pos_all = behavior_frames_metadata_df.set_index("behavior_frame_id")[
        ["x_pos_mm_interp", "y_pos_mm_interp"]
    ].to_numpy()  # (n_frames, 2={x,y})
    assert stage_pos_all.shape[0] == all_keypoint_pos_pixel_xy.shape[0], (
        "Number of frames in behavior frames metadata does not match number of "
        "frames in keypoint positions."
    )

    # Create calibration mapper
    logger.info("Creating Spotlight camera-stage-physical position mapper")
    cam_mapper = SpotlightPositionMapper(behavior_cam_calib_path)
    # Repeat stage pos along the keypoint axis of the 2D pose data
    n_frames, n_keypoints, _ = all_keypoint_pos_pixel_xy.shape
    stage_pos_all_rep = np.repeat(stage_pos_all[:, None, :], n_keypoints, axis=1)
    physical_pos_all = cam_mapper.stage_and_pixel_to_physical(
        stage_pos_all_rep, all_keypoint_pos_pixel_xy
    )
    assert physical_pos_all.shape == (n_frames, n_keypoints, 2)

    # Load recording configs to get arena size
    logger.info("Loading recording config to get arena size")
    with open(recorder_config_path, "r") as f:
        recorder_config = yaml.safe_load(f)
    arena_size_x_mm = recorder_config["arena"]["size_x_mm"]
    arena_size_y_mm = recorder_config["arena"]["size_y_mm"]
    xmin_allowd = edge_tolerance_mm
    xmax_allowd = arena_size_x_mm - edge_tolerance_mm
    ymin_allowd = edge_tolerance_mm
    ymax_allowd = arena_size_y_mm - edge_tolerance_mm
    logger.info(
        f"Arena size: x={arena_size_x_mm} mm, y={arena_size_y_mm} mm. "
        f"Edge tolerance: {edge_tolerance_mm} mm. "
        f"Allowed x range: [{xmin_allowd}, {xmax_allowd}]. "
        f"Allowed y range: [{ymin_allowd}, {ymax_allowd}]. "
    )

    # Determine if any keypoint is too close to edge
    logger.info("Detecting frames with keypoints too close to arena edge")
    too_close_flags = (
        (physical_pos_all[:, :, 0] < xmin_allowd)
        | (physical_pos_all[:, :, 0] > xmax_allowd)
        | (physical_pos_all[:, :, 1] < ymin_allowd)
        | (physical_pos_all[:, :, 1] > ymax_allowd)
    ).any(axis=1)
    too_close_df = pd.DataFrame()
    too_close_df["behavior_frameid"] = np.arange(n_frames, dtype="uint32")
    too_close_df["too_close_to_edge"] = too_close_flags

    return too_close_df
