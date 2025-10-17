import h5py
import numpy as np
import yaml
import json
import imageio.v2 as imageio
import logging
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm

from poseforge.spotlight.input_transform import (
    reverse_rotation_and_crop,
)
from poseforge.neuromechfly.constants import legs
from poseforge.pose.bodyseg.viz import draw_mask_contours


def process_spotlight_trial(
    bodyseg_output_path: Path,
    cropped_behavior_dir: Path,
    spotlight_dir: Path,
    visualize_masks: bool = True,
    viz_vmin: int = 200,
    viz_vmax: int = 1000,
    padding: int = 100,
):
    """
    Extract muscle traces from spotlight trial using bodyseg segmentation
    masks.

    TODO: REWRITE THIS NOTE
    Note:
        The directory `cropped_behavior_dir` contains images that are
        rotated and cropped from the original behavior images so that the
        fly is centered and aligned. The metadata for the cropping and
        rotation is stored in JSON files in the same directory. These
        rotated-and-cropped images have a size specified by
        `bodyseg_model_input_size` (e.g. 900x900).

        The predicted bodyseg masks have the same size as the input to the
        bodyseg model. The input to the bodyseg models is taken from
        `cropped_behavior_dir`, but it is resized to a smaller size (e.g.
        256x256) before being passed to the model. Therefore, upon loading
        bodyseg masks, the first thing to do is to resize them back to the
        `bodyseg_model_input_size`. Only then can we apply the reverse
        image transform to map the bodyseg masks back to the original
        behavior image space before rotation and cropping.

    Args:
        bodyseg_output_path: Path to bodyseg output H5 file.
        cropped_behavior_dir: Directory containing cropped behavior images
            and metadata.
        spotlight_dir: Directory containing spotlight muscle images.
        original_input_size: Original input size (height, width) used for
            bodyseg model training.
        visualize_masks: Whether to visualize the masks and save the
            outputs to a "muscle_roi_viz" directory under the directory
            where the outputs are saved.
    """
    muscle_frame_paths = {
        int(path.stem.split("_")[-1]): path
        for path in sorted(
            list(spotlight_dir.glob("processed/muscle_images/muscle_frame_*.tif"))
        )
    }

    # Load bodyseg masks and match to muscle frames
    segmap_by_muscle_frameid = {}
    metadata_path_by_muscle_frameid = {}
    print("Loading bodyseg output from:", bodyseg_output_path)
    with h5py.File(bodyseg_output_path, "r") as f_bodyseg:
        behavior_frame_ids = list(f_bodyseg["frame_ids"][:])
        pred_segmaps = f_bodyseg["pred_segmap"][:]
        seg_labels = list(f_bodyseg["pred_segmap"].attrs["class_labels"])
    print(f"Bodyseg output contains {len(behavior_frame_ids)} frames")

    # Get behavior-to-muscle frame ratio
    dual_recording_metadata_path = spotlight_dir / "metadata/dual_recording_timing.yaml"
    with open(dual_recording_metadata_path, "r") as f:
        dual_recording_metadata = yaml.safe_load(f)
    behavior_to_muscle_ratio = dual_recording_metadata["sync_ratio"]

    # Match muscle frames to behavior frames and load corresponding segmaps
    not_found_count = 0
    for muscle_frame_id in muscle_frame_paths.keys():
        behavior_frame_id = (muscle_frame_id + 1) * behavior_to_muscle_ratio
        if behavior_frame_id not in behavior_frame_ids:
            not_found_count += 1
            continue
        segmap = pred_segmaps[behavior_frame_ids.index(behavior_frame_id)]
        segmap_by_muscle_frameid[muscle_frame_id] = segmap
        transform_metadata_path = (
            cropped_behavior_dir.parent.parent
            / "all"
            / f"frame_{behavior_frame_id:09d}.metadata.json"
        )
        metadata_path_by_muscle_frameid[muscle_frame_id] = transform_metadata_path

    logging.info(
        f"Found {len(segmap_by_muscle_frameid)} muscle frames with corresponding "
        f"bodyseg masks. {not_found_count} muscle frames did not have corresponding "
        f"bodyseg masks."
    )

    out_dir = bodyseg_output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if visualize_masks:
        viz_out_dir = out_dir / "muscle_roi_viz"
        viz_out_dir.mkdir(parents=True, exist_ok=True)
        segs_to_visualize_names = [
            x for x in seg_labels if x[:2] in legs or x.endswith("Antenna")
        ]
        segs_to_visualize_indices = [
            i for i, x in enumerate(seg_labels) if x in segs_to_visualize_names
        ]

    # Process each muscle frame
    muscle_frame_ids = list(segmap_by_muscle_frameid.keys())
    mapped_data = []

    print("Processing muscle frames:", len(muscle_frame_ids))
    for muscle_frame_id in tqdm(muscle_frame_ids, disable=None):
        segmap = segmap_by_muscle_frameid[muscle_frame_id]

        # Load muscle image
        # The postprocessing step in spotlight-tools has already warped muscle images to
        # the corresponding behavior images. So just load them here.
        muscle_image_path = muscle_frame_paths[muscle_frame_id]
        muscle_image = imageio.imread(muscle_image_path)

        # Reverse-transform segmap to original space
        # (before alignment via rotation and cropping)
        transform_metadata_path = metadata_path_by_muscle_frameid[muscle_frame_id]
        with open(transform_metadata_path, "r") as f:
            crop_metadata = json.load(f)
        behavior_frame_id = (muscle_frame_id + 1) * behavior_to_muscle_ratio
        assert crop_metadata["frame_id"] == behavior_frame_id
        size_before_transform = tuple(crop_metadata["rotation"]["input_size"])
        if (
            size_before_transform != muscle_image.shape
            and muscle_frame_id == muscle_frame_ids[0]
        ):
            logging.warning(
                f"Muscle image shape {muscle_image.shape} does not match target (behavior) "
                f"shape {size_before_transform}. This is likely because the behavior camera "
                f"cannot record at any arbitrary frame size, and it is rounded to the "
                f"nearest allowable size (multiples of 64). Cropping the muscle image "
                f"to the target size."
            )
            assert size_before_transform[0] <= muscle_image.shape[0]
            assert size_before_transform[1] <= muscle_image.shape[1]
            muscle_image = muscle_image[
                : size_before_transform[0], : size_before_transform[1]
            ]

        # The bodyseg pipeline (and pose pipelines in general) resizes the input to a
        # smaller working dimension (e.g. 256x256) before feeding it to the model. The
        # model predictions (e.g. segmap) are also produced at this smaller size.
        # Therefore, first resize the output back to the dimensions immediately after
        # alignment and before cropping by pose pipelines.
        size_after_crop = crop_metadata["crop"]["output_size"]
        zoom_factor = (
            size_after_crop[0] / segmap.shape[0],
            size_after_crop[1] / segmap.shape[1],
        )
        segmap_before_resize = ndimage.zoom(segmap, zoom_factor, order=0)

        # Undo alignment transformations (rotation + crop) to map segmap back to the
        # pixel space of raw Spotlight behavior images
        segmap_before_alignment = reverse_rotation_and_crop(
            segmap_before_resize,
            rotation_params=crop_metadata["rotation"],
            crop_params=crop_metadata["crop"],
            fill_value=0,
        )

        # Crop muscle image and segmap to a rough bounding box
        ys, xs = np.where(segmap_before_alignment > 0)
        feature_row_min = ys.min()
        feature_row_max = ys.max()
        feature_col_min = xs.min()
        feature_col_max = xs.max()
        crop_row_min = max(0, feature_row_min - padding)
        crop_row_max = min(muscle_image.shape[0], feature_row_max + padding)
        crop_col_min = max(0, feature_col_min - padding)
        crop_col_max = min(muscle_image.shape[1], feature_col_max + padding)
        muscle_image_cropped = muscle_image[
            crop_row_min:crop_row_max, crop_col_min:crop_col_max
        ]
        segmap_cropped = segmap_before_alignment[
            crop_row_min:crop_row_max, crop_col_min:crop_col_max
        ]
        cropping_metadata = {
            "segmap_feature_bbox_row_range": [feature_row_min, feature_row_max],
            "segmap_feature_bbox_col_range": [feature_col_min, feature_col_max],
            "crop_box_row_range": [crop_row_min, crop_row_max],
            "crop_box_col_range": [crop_col_min, crop_col_max],
        }

        mapping_results = {
            "muscle_frame_id": muscle_frame_id,
            "muscle_image_path": str(muscle_image_path.absolute()),
            "input_transform_metadata_path": str(transform_metadata_path.absolute()),
            "muscle_image_cropped": muscle_image_cropped,
            "segmap_cropped": segmap_cropped,
            "cropping_metadata": cropping_metadata,
        }
        mapped_data.append(mapping_results)

        if visualize_masks:
            masks = []
            for seg_value in segs_to_visualize_indices:
                masks.append(segmap_cropped == seg_value)
            draw_mask_contours(
                image=muscle_image_cropped,
                masks=np.stack(masks, axis=0),
                vmin=viz_vmin,
                vmax=viz_vmax,
                output_path=viz_out_dir / f"frame_{muscle_frame_id:06d}.png",
            )

    with h5py.File(out_dir / "mapped_segmaps_and_muscle_images.h5", "w") as f_out:
        for data in mapped_data:
            grp = f_out.create_group(f"muscle_frame_{data['muscle_frame_id']:06d}")
            grp.create_dataset(
                "muscle_image_cropped",
                data=data["muscle_image_cropped"],
                compression="gzip",
                shuffle=True,
            )
            grp.create_dataset(
                "segmap_cropped",
                data=data["segmap_cropped"],
                compression="gzip",
                shuffle=True,
            )
            grp.attrs["muscle_image_path"] = data["muscle_image_path"]
            grp.attrs["input_transform_metadata_path"] = data[
                "input_transform_metadata_path"
            ]
            grp.attrs.update(data["cropping_metadata"])

    return segmap_by_muscle_frameid


if __name__ == "__main__":
    bodyseg_model_dir = Path("bulk_data/pose_estimation/bodyseg/trial_20251012b/")
    spotlight_basedir = Path("bulk_data/spotlight_recordings/")
    cropped_behavior_basedir = Path(
        "bulk_data/behavior_images/spotlight_aligned_and_cropped/"
    )

    spotlight_trial = "20250613-fly1b-005"
    bodyseg_output_path = (
        bodyseg_model_dir
        / f"inference/{spotlight_trial}_model_prediction_not_flipped/bodyseg_pred.h5"
    )
    spotlight_dir = spotlight_basedir / spotlight_trial
    cropped_behavior_dir = (
        cropped_behavior_basedir / spotlight_trial / "model_prediction/not_flipped"
    )

    process_spotlight_trial(
        bodyseg_output_path,
        cropped_behavior_dir,
        spotlight_dir,
        visualize_masks=True,
    )
