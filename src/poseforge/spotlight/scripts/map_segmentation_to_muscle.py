import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from pathlib import Path

from poseforge.spotlight.muscle_segmentation import MuscleSegmentationPipeline


if __name__ == "__main__":
    bodyseg_model_dir = Path("bulk_data/pose_estimation/bodyseg/trial_20251012b/")
    spotlight_basedir = Path("bulk_data/spotlight_recordings/")
    cropped_behavior_basedir = Path(
        "bulk_data/behavior_images/spotlight_aligned_and_cropped/"
    )
    muscle_vmin = 200
    muscle_vmax = 1000

    for spotlight_dir in spotlight_basedir.glob("20250613-fly1b-012"):
        spotlight_trial = spotlight_dir.stem
        bodyseg_output_path = (
            bodyseg_model_dir
            / f"inference/{spotlight_trial}_model_prediction_not_flipped/bodyseg_pred.h5"
        )
        cropped_behavior_dir = (
            cropped_behavior_basedir / spotlight_trial / "model_prediction/not_flipped"
        )
        muscle_traces_dir = spotlight_dir / "muscle_traces"
        output_path = muscle_traces_dir / "muscle_mapping.h5"
        debug_plots_dir = muscle_traces_dir / "muscle_mapping_debug_plots"

        logging.info(
            f"Processing spotlight trial: {spotlight_trial}; "
            f"saving output to {output_path}"
        )

        pipeline = MuscleSegmentationPipeline(
            spotlight_trial_dir=spotlight_dir,
            aligned_behavior_image_dir=cropped_behavior_dir,
            bodyseg_prediction_path=bodyseg_output_path,
            output_path=output_path,
            muscle_vrange=(muscle_vmin, muscle_vmax),
            debug_plots_dir=debug_plots_dir,
        )
        pipeline.apply_initial_mapping(n_workers=-1)
        pipeline.apply_fine_alignment(n_workers=-1)
        pipeline.apply_morph_denoising(n_workers=-1)
        pipeline.apply_mask_dilation(dilation_size=7, n_workers=-1)
