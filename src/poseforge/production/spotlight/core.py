import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import h5py
import yaml
import logging
from time import perf_counter
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

from spotlight_tools.calibration import SpotlightPositionMapper

logging.getLogger("pvio").setLevel(logging.INFO)
from pvio.torch_tools import SimpleVideoCollectionLoader

import poseforge.spotlight.flip_detection as flip_detection
import poseforge.pose.bodyseg as bodyseg
import poseforge.pose.keypoints3d as keypoints3d
from poseforge.production.config import load_config
from poseforge.pose.camera import CameraToWorldMapper
from poseforge.neuromechfly.constants import keypoint_segments_canonical


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
        self.completed_steps = {
            "detect_usable_frames": False,
            "predict_keypoints3d": False,
            "predict_body_segmentation": False,
            "visualize_keypoints3d": False,
            "visualize_body_segmentation": False,
        }

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
        logger.info("Detecting usable frames based on flip and edge criteria")
        df_flipped = self._detect_flipped_frames(
            loading_batch_size=loading_batch_size,
            loading_n_workers=loading_n_workers,
            loading_buffer_size=loading_buffer_size,
            loading_cache_video_metadata=loading_cache_video_metadata,
        )
        df_edge = self._detect_close_to_edge(edge_tolerance_mm=edge_tolerance_mm)
        df = pd.merge(df_flipped, df_edge, on="behavior_frameid", how="inner")

        is_not_flipped = df["flip_detection"] == "not_flipped"
        is_not_too_close = ~df["too_close_to_edge"]
        df["usable"] = is_not_flipped & is_not_too_close
        df.to_csv(self.paths.usable_frames, index=False)
        logger.info(
            f"Out of {len(df)} frames, {is_not_flipped.sum()} frames are not flipped, "
            f"{is_not_too_close.sum()} frames are not too close to edge, and "
            f"{df['usable'].sum()} frames meet both criteria."
            f"Saved usable frames metadata to {self.paths.usable_frames}"
        )
        self.completed_steps["detect_usable_frames"] = True
        return df

    def _detect_flipped_frames(
        self,
        loading_batch_size: int = 128,
        loading_n_workers: int = 8,
        loading_buffer_size: int = 128,
        loading_cache_video_metadata: bool = True,
    ) -> pd.DataFrame:
        # Set up flip detection model
        logger.info("Setting up flip detection model")
        model = flip_detection.model.FlipDetectionCNN(
            **self.model_config["flip_detection"]["init_params"]
        )
        checkpoint_path = Path(
            self.model_config["flip_detection"]["checkpoint"]
        ).expanduser()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        class_labels = np.array(self.model_config["flip_detection"]["output_labels"])

        # Create video loader
        logger.info("Creating video loader for flip detection")
        working_size = self.model_config["flip_detection"]["working_size"]
        transform_ = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((working_size, working_size)),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        video_loader = SimpleVideoCollectionLoader(
            [self.paths.aligned_behavior_video],
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
                logits = model(frames.to(self.device))
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

    def _detect_close_to_edge(self, edge_tolerance_mm: float) -> pd.DataFrame:
        # Load keypoint positions predicted by 2D pose model
        logger.info("Loading keypoint positions from 2D pose model")
        with h5py.File(self.paths.alignment_metadata, "r") as f:
            ds = f["keypoints_xy_pre_alignment"]
            keypoints = ds.attrs["keypoint_names"]
            keypoints_exp = ["neck", "thorax", "abdomen tip"]
            assert keypoints.tolist() == keypoints_exp, "Unexpected keypoint names"
            all_keypoint_pos_pixel_xy = ds[:]  # (n_frames, num_keypoints=3, 2={x,y})

        # Load stage positions
        logger.info("Loading stage positions from behavior frames metadata")
        behavior_frames_metadata_df = pd.read_csv(self.paths.behavior_frames_metadata)
        stage_pos_all = behavior_frames_metadata_df.set_index("behavior_frame_id")[
            ["x_pos_mm_interp", "y_pos_mm_interp"]
        ].to_numpy()  # (n_frames, 2={x,y})
        assert stage_pos_all.shape[0] == all_keypoint_pos_pixel_xy.shape[0], (
            "Number of frames in behavior frames metadata does not match number of "
            "frames in keypoint positions."
        )

        # Create calibration mapper
        logger.info("Creating Spotlight camera-stage-physical position mapper")
        cam_mapper = SpotlightPositionMapper(self.paths.behavior_cam_calib)
        # Repeat stage pos along the keypoint axis of the 2D pose data
        n_frames, n_keypoints, _ = all_keypoint_pos_pixel_xy.shape
        stage_pos_all_rep = np.repeat(stage_pos_all[:, None, :], n_keypoints, axis=1)
        physical_pos_all = cam_mapper.stage_and_pixel_to_physical(
            stage_pos_all_rep, all_keypoint_pos_pixel_xy
        )
        assert physical_pos_all.shape == (n_frames, n_keypoints, 2)

        # Load recording configs to get arena size
        logger.info("Loading recording config to get arena size")
        with open(self.paths.recorder_config, "r") as f:
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
    ) -> Path:
        logger.info("Estimating 3D keypoint positions for behavior video")

        # Set up model and pipeline
        model_info = self.model_config["keypoints3d"]
        architecture_config_path = Path(model_info["architecture_config"]).expanduser()
        logger.info(
            f"Setting up 3D keypoints model from architecture config "
            f"{architecture_config_path}"
        )
        model = keypoints3d.Pose2p5DModel.create_architecture_from_config(
            architecture_config_path
        ).cuda()
        ckpt_path = Path(model_info["checkpoint"]).expanduser()
        logger.info(f"Loading 3D keypoints model weights from {ckpt_path}")
        weights_config = keypoints3d.config.ModelWeightsConfig(model_weights=ckpt_path)
        model.load_weights_from_config(weights_config)
        logger.info("Creating 3D keypoints inference pipeline")
        pipeline = keypoints3d.Pose2p5DPipeline(
            model, device=self.device, use_float16=True
        )

        # Set up camera mapper
        cam_mapper = CameraToWorldMapper(
            camera_pos, camera_fov_deg, camera_rendering_size, camera_rotation_euler
        )

        # Create video loader
        logger.info("Creating video loader for 3D keypoints prediction")
        working_size = model_info["working_size"]
        video_loader = SimpleVideoCollectionLoader(
            [self.paths.aligned_behavior_video],
            transform=transforms.Resize((working_size, working_size)),
            batch_size=loading_batch_size,
            num_workers=loading_n_workers,
            buffer_size=loading_buffer_size,
            use_cached_video_metadata=loading_cache_video_metadata,
        )

        # Create output file
        logger.info(
            f"Creating H5 file for 3D keypoint position predictions: "
            f"{self.paths.keypoints3d_prediction}"
        )
        n_frames_total = len(video_loader.dataset)
        with h5py.File(self.paths.keypoints3d_prediction, "w") as f:
            keypoint_names = model_info["keypoint_names"]
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
                logger.debug(
                    f"Saved output for {len(frame_ids)} frames in {elapsed:.3f}s"
                )

        self.completed_steps["predict_keypoints3d"] = True
        return self.paths.keypoints3d_prediction

    def predict_body_segmentation(
        self,
        loading_batch_size: int = 128,
        loading_n_workers: int = 8,
        loading_buffer_size: int = 128,
        loading_cache_video_metadata: bool = True,
    ) -> Path:
        logger.info("Estimating body segmentation masks for behavior video")

        # Set up model and pipeline
        model_info = self.model_config["bodyseg"]
        architecture_config_path = Path(model_info["architecture_config"]).expanduser()
        logger.info(
            f"Setting up body segmentation model from architecture config "
            f"{architecture_config_path}"
        )
        model = bodyseg.BodySegmentationModel.create_architecture_from_config(
            architecture_config_path
        ).cuda()
        ckpt_path = Path(model_info["checkpoint"]).expanduser()
        logger.info(f"Loading body segmentation model weights from {ckpt_path}")
        weights_config = bodyseg.config.ModelWeightsConfig(model_weights=ckpt_path)
        model.load_weights_from_config(weights_config)
        logger.info("Creating body segmentation inference pipeline")
        pipeline = bodyseg.BodySegmentationPipeline(
            model, device=self.device, use_float16=True
        )

        # Create video loader
        logger.info("Creating video loader for body segmentation")
        working_size = model_info["working_size"]
        video_loader = SimpleVideoCollectionLoader(
            [self.paths.aligned_behavior_video],
            transform=transforms.Resize((working_size, working_size)),
            batch_size=loading_batch_size,
            num_workers=loading_n_workers,
            buffer_size=loading_buffer_size,
            use_cached_video_metadata=loading_cache_video_metadata,
        )

        # Create output file
        logger.info(
            f"Creating H5 file for bodyseg predictions: {self.paths.bodyseg_prediction}"
        )
        n_frames_total = len(video_loader.dataset)
        with h5py.File(self.paths.bodyseg_prediction, "w") as f:
            ds_confidence = f.create_dataset(
                "confidence",
                shape=(n_frames_total, working_size, working_size),
                dtype="uint8",
                compression="gzip",
            )
            ds_labels = f.create_dataset(
                "labels",
                shape=(n_frames_total, working_size, working_size),
                dtype="uint8",
                compression="gzip",
            )
            f.attrs["class_labels"] = pipeline.class_labels

            # Run inference
            logger.info("Running inference for body segmentation")
            for batch in tqdm(
                video_loader, desc="Predicting bodyseg", unit="batch", disable=None
            ):
                # Forward pass
                # No need to move data to GPU and back, Pipeline.inference handles that
                frames = batch["frames"]
                frame_ids = batch["frame_indices"]
                assert (np.array(batch["video_indices"]) == 0).all()
                pred_dict = pipeline.inference(frames)
                logits = pred_dict["logits"]  # (B, n_classes, H, W)
                confidence = pred_dict["confidence"]  # (B, H, W)
                confidence = (confidence * 100).to(torch.uint8)
                labels = torch.argmax(logits, dim=1).to(torch.uint8)  # (B, H, W)

                # Save to H5 file
                start_time = perf_counter()
                ds_labels[frame_ids, :, :] = labels.numpy()
                ds_confidence[frame_ids, :, :] = confidence.numpy()
                elapsed = perf_counter() - start_time
                logger.debug(
                    f"Saved output for {len(frame_ids)} frames in {elapsed:.3f}s"
                )
        self.completed_steps["predict_body_segmentation"] = True
        return self.paths.bodyseg_prediction

    def visualize_bodyseg_predictions(self):
        pass


@dataclass
class SpotlightRecordingPaths:
    """Dataclass containing all file paths for a Spotlight recording with validation.

    This class handles the initialization of all file paths required for processing
    a Spotlight recording and validates that the required files exist. It provides
    a clean separation between path management and the main processing logic.

    Attributes:
        data_path: Base path to the recording data directory
        with_muscle: Whether muscle data is expected to be available
        muscle_available: Whether muscle data files are actually available (set after
            validation)
        (see below for automatically derived paths)

    Paths to metadata generated by the Spotlight recorder (not including metadata
    generated during postprocessing):
        - recorder_metadata_dir (dir):
            Base directory containing such metadata files
        - recorder_config (YAML):
            `recorder_config.yaml` file used by the Spotlight recorder for this trial
        - behavior_cam_calib (YAML):
            Behavior camera calibration parameters (i.e. affine transforms for mapping
            between behavior camera, stage, and physical coords)
        - experiment_param (YAML):
            Settings set in the Spotlight recorder GUI
        - dual_recording_timing_config (YAML, only if with muscle):
            Information on synchronization between behavior and muscle cameras
        - muscle_cam_calib (YAML, only if with muscle):
            Muscle camera calibration parameters (i.e. affine transforms for mapping
            between muscle camera, stage, and physical coords)

    Paths to processed data generated by `spotlight_tools.postprocessing`:
        - processed_dir (dir):
            Base directory containing such processed data
        - aligned_behavior_video (MKV):
            Aligned behavior video file
        - alignment_metadata (HDF5):
            Parameters for transforms applied to align the fly, including simple
            2D pose predicted by a SLEAP model (3 keypoints only). This was used to
            determine the fly's orientation and position, which were then used to align
            the frames
        - behavior_frames_metadata (CSV):
            Metadata for each behavior frame, including timing information (acquisition
            time, etc.) and stage positions at the time of each behavior frame
            interpolated from the raw stage tracking log
        - aligned_muscle_frames_dir (dir, only if with muscle):
            Directory containing aligned muscle images (individual TIFF files)
        - muscle_frames_metadata (CSV, only if with muscle):
            Metadata for each muscle frame, including timing information (acquisition
            time, etc.) and stage positions at the time of each muscle frame
            interpolated from the raw stage tracking log

    Paths to output files to be generated by the poseforge production pipeline:
        - output_dir (dir):
            Base directory for such output files
        - usable_frames (CSV):
            Metadata on which behavior frames are usable based on (1) whether the fly
            is flipped and (2) whether any keypoint is too close to the arena edge.
        - keypoints3d_prediction (HDF5):
            3D keypoint model predictions
        - invkin_output (HDF5):
            Inverse kinematics solution and forward kinematics reconstructions
        - bodyseg_prediction (HDF5):
            Body segmentation model predictions
        - keypoints3d_viz (MP4):
            Visualization video for 3D keypoint predictions, including inverse and
            forward kinematics
        - bodyseg_viz (MP4):
            Visualization video for body segmentation predictions

    Raises:
        FileNotFoundError: If required files are missing during validation
    """

    trial_basedir: Path
    with_muscle: bool = False

    def __post_init__(self):
        """Initialize derived paths and validate required files."""
        # Convert to Path if string
        self.trial_basedir = Path(self.trial_basedir)

        # Metadata paths - from Spotlight recorder
        self.recorder_metadata_dir = self.trial_basedir / "metadata/"
        _metadir = self.recorder_metadata_dir
        self.recorder_config = _metadir / "recorder_config.yaml"
        self.behavior_cam_calib = _metadir / "calibration_parameters_behavior.yaml"
        self.experiment_param = _metadir / "experiment_parameters.yaml"
        if self.with_muscle:
            self.dual_recording_timing_config = _metadir / "dual_recording_timing.yaml"
            self.muscle_cam_calib = _metadir / "calibration_parameters_muscle.yaml"
        else:
            self.dual_recording_timing_config = None
            self.muscle_cam_calib = None

        # Processed paths - from spotlight_tools.postprocessing
        self.processed_dir = self.trial_basedir / "processed/"
        _procdir = self.processed_dir
        self.aligned_behavior_video = _procdir / "aligned_behavior_video.mkv"
        self.alignment_metadata = _procdir / "behavior_alignment_transforms.h5"
        self.behavior_frames_metadata = _procdir / "behavior_frames_metadata.csv"
        if self.with_muscle:
            self.aligned_muscle_frames_dir = _procdir / "aligned_muscle_images/"
            self.muscle_frames_metadata = _procdir / "muscle_frames_metadata.csv"
        else:
            self.aligned_muscle_frames_dir = None
            self.muscle_frames_metadata = None

        # Check if required files exist
        required_behavior_path_attrs = [
            "recorder_config",
            "behavior_cam_calib",
            "experiment_param",
            "aligned_behavior_video",
            "alignment_metadata",
            "behavior_frames_metadata",
        ]
        required_muscle_path_attrs = [
            "dual_recording_timing_config",
            "muscle_cam_calib",
            "aligned_muscle_frames_dir",
            "muscle_frames_metadata",
        ]
        if not self._check_paths_exist(required_behavior_path_attrs):
            raise FileNotFoundError(
                "One or more generally required data files are missing."
            )
        if self.with_muscle and not self._check_paths_exist(required_muscle_path_attrs):
            raise FileNotFoundError(
                "with_muscle is set to True but one or more required muscle data files "
                "are missing."
            )

        # Output - to be generated by this module
        self.output_dir = self.trial_basedir / "poseforge_output/"
        _outdir = self.output_dir
        _outdir.mkdir(parents=True, exist_ok=True)
        self.usable_frames = _outdir / "usable_frames.csv"
        self.bodyseg_prediction = _outdir / "bodyseg_prediction.h5"
        self.keypoints3d_prediction = _outdir / "keypoints3d_prediction.h5"
        self.invkin_output = _outdir / "inverse_kinematics_output.h5"

    def _check_paths_exist(self, required_path_attrs: list[str]) -> bool:
        for path_attr in required_path_attrs:
            if not hasattr(self, path_attr):
                raise AttributeError(
                    f"Input object does not have attribute '{path_attr}'"
                )
            path = getattr(self, path_attr)
            if not path.exists():
                logger.error(f"Required file '{path_attr}' doesn't exist: {path}")
                return False
        return True


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
    recording.predict_body_segmentation(loading_n_workers=8)
    recording.predict_keypoints3d(loading_n_workers=8)
