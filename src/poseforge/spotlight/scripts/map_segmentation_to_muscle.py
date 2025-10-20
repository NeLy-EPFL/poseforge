import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from pathlib import Path

from poseforge.spotlight.muscle_segmentation import process_muscle_segmentation


if __name__ == "__main__":
    bodyseg_model_dir = Path("bulk_data/pose_estimation/bodyseg/trial_20251012b/")
    spotlight_basedir = Path("bulk_data/spotlight_recordings/")
    cropped_behavior_basedir = Path(
        "bulk_data/behavior_images/spotlight_aligned_and_cropped/"
    )
    muscle_vmin = 200
    muscle_vmax = 1000

    for spotlight_dir in spotlight_basedir.glob("20250613-fly1b-*"):
        spotlight_trial = spotlight_dir.stem
        bodyseg_output_path = (
            bodyseg_model_dir
            / f"inference/{spotlight_trial}_model_prediction_not_flipped/bodyseg_pred.h5"
        )
        cropped_behavior_dir = (
            cropped_behavior_basedir / spotlight_trial / "model_prediction/not_flipped"
        )
        muscle_traces_dir = spotlight_dir / "muscle_traces"
        output_path = muscle_traces_dir / "muscle_mapping_refactored.h5"
        debug_plots_dir = muscle_traces_dir / "muscle_mapping_debug_plots_refactored"

        logging.info(
            f"Processing spotlight trial: {spotlight_trial}; "
            f"saving output to {output_path}"
        )

        process_muscle_segmentation(
            spotlight_trial_dir=spotlight_dir,
            aligned_behavior_image_dir=cropped_behavior_dir,
            bodyseg_prediction_path=bodyseg_output_path,
            output_path=output_path,
            muscle_vrange=(muscle_vmin, muscle_vmax),
            padding=100,
            foreground_classes="legs",
            search_limit=50,
            morph_kernel_size=5,
            morph_iterations=2,
            dilation_size=7,
            debug_plots_dir=debug_plots_dir,
            n_workers=-1,
        )
