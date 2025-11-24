from loguru import logger
from poseforge.util.sys import set_loguru_level

set_loguru_level("INFO")

from pathlib import Path
from importlib.resources import files

from poseforge.production.spotlight.core import SpotlightRecordingProcessor


# Load model configurations
model_config_path = files("poseforge.production.spotlight").joinpath("config.yaml")
if not model_config_path.exists():
    raise FileNotFoundError(
        f"Model config file not found at expected location: {model_config_path}"
    )
logger.info(f"Using model configurations from {model_config_path}")

# Specify recordings to process
spotlight_data_basedir = Path("~/data/spotlight/").expanduser()
spotlight_recording_dirs = sorted(spotlight_data_basedir.glob("20250613-fly1b-*"))

# Process each recording
for trial_dir in spotlight_recording_dirs:
    logger.info(f"Processing spotlight recording at {trial_dir}")
    recording = SpotlightRecordingProcessor(
        trial_dir, model_config_path, with_muscle=True
    )
    recording.detect_usable_frames(edge_tolerance_mm=5.0, loading_n_workers=8)
    recording.predict_keypoints3d(loading_n_workers=8)
    recording.solve_inverse_kinematics()
    recording.visualize_keypoints3d()
    recording.predict_body_segmentation(loading_n_workers=8)
    recording.visualize_bodyseg_predictions()
