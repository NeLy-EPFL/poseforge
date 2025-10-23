from pathlib import Path

from poseforge.spotlight.muscle_segmentation import (
    process_muscle_segmentation,
    leg_segment_names,
)
from poseforge.spotlight.viz import plot_muscle_traces_with_kinematics
from poseforge.util.sys import setup_logger


if __name__ == "__main__":
    logger = setup_logger(level="info")

    bodyseg_model_dir = Path("bulk_data/pose_estimation/bodyseg/trial_20251012b/")
    keypoints3d_model_dir = Path(
        "bulk_data/pose_estimation/keypoints3d/trial_20251013b/"
    )
    spotlight_basedir = Path("bulk_data/spotlight_recordings/")
    cropped_behavior_basedir = Path(
        "bulk_data/behavior_images/spotlight_aligned_and_cropped/"
    )
    muscle_vrange = (200, 1000)

    for spotlight_dir in sorted(spotlight_basedir.glob("20250613-fly1b-013")):
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
        keypoints3d_prediction_path = (
            keypoints3d_model_dir
            / f"production/epoch14/{spotlight_trial}/keypoints3d.h5"
        )
        inverse_kinematics_data_path = (
            keypoints3d_model_dir
            / f"production/epoch14/{spotlight_trial}/inverse_kinematics.h5"
        )

        skipped_trials = []
        if not bodyseg_output_path.is_file():
            logger.error(
                f"Bodyseg output file not found for trial {spotlight_trial}: "
                f"{bodyseg_output_path}; skipping"
            )
            skipped_trials.append(spotlight_trial)
            continue

        logger.info(
            f"Processing spotlight trial: {spotlight_trial}; "
            f"saving output to {output_path}"
        )
        process_muscle_segmentation(
            spotlight_trial_dir=spotlight_dir,
            preprocessed_behavior_image_dir=cropped_behavior_dir,
            bodyseg_prediction_path=bodyseg_output_path,
            output_path=output_path,
            muscle_traces_segments=leg_segment_names,
            dilation_kernel_size=7,
            muscle_vrange=muscle_vrange,
            debug_plots_dir=debug_plots_dir,
            n_workers=-2,
        )

        viz_trange = (161, 165)
        plot_muscle_traces_with_kinematics(
            muscle_segmentation_data_path=output_path,
            keypoints3d_data_path=keypoints3d_prediction_path,
            inverse_kinematics_data_path=inverse_kinematics_data_path,
            muscle_segments_to_plot=["Coxa", "Femur", "Tibia"],
            output_path=muscle_traces_dir / "muscle_traces_with_kinematics.pdf",
            trange=viz_trange,
            title=f"{spotlight_trial} ({viz_trange[0]}–{viz_trange[1]} s)",
            # display=True,
        )

    if len(skipped_trials) > 0:
        logger.error(f"Skipped {len(skipped_trials)} trials: {skipped_trials}")
