import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from pathlib import Path

from poseforge.spotlight.muscle_segmentation import (
    process_muscle_segmentation,
    extract_muscle_trace,
    leg_segment_names,
)


if __name__ == "__main__":
    bodyseg_model_dir = Path("bulk_data/pose_estimation/bodyseg/trial_20251012b/")
    spotlight_basedir = Path("bulk_data/spotlight_recordings/")
    cropped_behavior_basedir = Path(
        "bulk_data/behavior_images/spotlight_aligned_and_cropped/"
    )
    muscle_vrange = (200, 1000)

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
        output_path = muscle_traces_dir / "muscle_mapping_refactored.h5"
        debug_plots_dir = muscle_traces_dir / "muscle_mapping_debug_plots_refactored"

        logging.info(
            f"Processing spotlight trial: {spotlight_trial}; "
            f"saving output to {output_path}"
        )

        # process_muscle_segmentation(
        #     spotlight_trial_dir=spotlight_dir,
        #     aligned_behavior_image_dir=cropped_behavior_dir,
        #     bodyseg_prediction_path=bodyseg_output_path,
        #     output_path=output_path,
        #     foreground_classes_for_alignment=leg_segment_names,
        #     muscle_vrange=muscle_vrange,
        #     padding=100,
        #     search_limit=50,
        #     morph_kernel_size=5,
        #     morph_iterations=1,
        #     dilation_size=7,
        #     debug_plots_dir=debug_plots_dir,
        #     n_workers=-1,
        # )
        muscle_traces, muscle_pixel_values = extract_muscle_trace(
            muscle_segmentation_path=output_path,
            body_segments=leg_segment_names,
            use_dilated=True,
        )

        import matplotlib.pyplot as plt
        import numpy as np

        fig, axs = plt.subplots(
            len(leg_segment_names),
            2,
            figsize=(12, 2 * len(leg_segment_names)),
            tight_layout=True,
            sharex=True,
        )
        for i, label in enumerate(leg_segment_names):
            # Raw fluorescence
            ax = axs[i, 0]
            frame_ids = sorted(muscle_traces[label].keys())
            activations = [muscle_traces[label][fid] for fid in frame_ids]
            times = [fid / 30 for fid in frame_ids]
            ax.plot(times, activations, label=label)
            # ax.set_ylim(*muscle_vrange)
            ax.set_ylabel(label, rotation=0, labelpad=30)
            ax.set_yticks([300, 900])
            if i == 0:
                ax.set_title("Fluorescence")
            if i == len(leg_segment_names) - 1:
                ax.set_xlabel("Time (s)")
            
            # DF/F
            f0 = np.nanpercentile(activations, 10)
            df_f = (activations - f0) / f0
            ax = axs[i, 1]
            ax.plot(times, df_f, label=label)
            # ax.set_ylim(-0.3, 1.3)
            ax.set_yticks([0, 1])
            if i == 0:
                ax.set_title(r"$\Delta$F/F")
            if i == len(leg_segment_names) - 1:
                ax.set_xlabel("Time (s)")

        plt.show()