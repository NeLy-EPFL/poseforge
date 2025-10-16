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
    rotate_image_around_point,
    crop_image_and_keypoints,
)
from poseforge.neuromechfly.constants import legs
from poseforge.pose.bodyseg.viz import draw_mask_contours


def process_spotlight_trial(
    bodyseg_output_path: Path,
    cropped_behavior_dir: Path,
    spotlight_dir: Path,
    original_input_size: tuple[int, int],
    visualize_masks: bool = True,
    viz_vmin: int = 200,
    viz_vmax: int = 1000,
):
    """
    Extract muscle traces from spotlight trial using bodyseg segmentation
    masks.

    Note:
        The directory `cropped_behavior_dir` contains images that are
        rotated and cropped from the original behavior images so that the
        fly is centered and aligned. The metadata for the cropping and
        rotation is stored in JSON files in the same directory. These
        rotated-and-cropped images have a size specified by
        `original_input_size` (e.g. 900x900).

        The predicted bodyseg masks have the same size as the input to the
        bodyseg model. The input to the bodyseg models is taken from
        `cropped_behavior_dir`, but it is resized to a smaller size (e.g.
        256x256) before being passed to the model. Therefore, upon loading
        bodyseg masks, the first thing to do is to resize them back to the
        `original_input_size`. Only then can we apply the reverse image
        transform to map the bodyseg masks back to the original behavior
        image space before rotation and cropping.

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
    crop_metadata_path_by_muscle_frameid = {}
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
        crop_metadata_path = (
            cropped_behavior_dir.parent.parent
            / "all"
            / f"frame_{behavior_frame_id:09d}.metadata.json"
        )
        crop_metadata_path_by_muscle_frameid[muscle_frame_id] = crop_metadata_path

    logging.info(
        f"Found {len(segmap_by_muscle_frameid)} muscle frames with corresponding "
        f"bodyseg masks. {not_found_count} muscle frames did not have corresponding "
        f"bodyseg masks."
    )

    out_dir = bodyseg_output_path.parent / "muscle_segmentation"
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
    print("Processing muscle frames:", len(muscle_frame_ids))
    for muscle_frame_id in tqdm(muscle_frame_ids, disable=None):
        segmap = segmap_by_muscle_frameid[muscle_frame_id]
        # Resize segmap to original_input_size using nearest neighbor sampling
        if segmap.shape != original_input_size:
            # Calculate zoom factors for each dimension
            zoom_factors = (
                original_input_size[0] / segmap.shape[0],
                original_input_size[1] / segmap.shape[1],
            )
            # Resize using nearest neighbor interpolation (order=0)
            segmap = ndimage.zoom(segmap, zoom_factors, order=0)
            assert segmap.shape == original_input_size

        # Load muscle frame
        muscle_image_path = muscle_frame_paths[muscle_frame_id]
        muscle_image = imageio.imread(muscle_image_path)

        # Rotate and crop the muscle image the same way as the behavior image
        crop_metadata_path = crop_metadata_path_by_muscle_frameid[muscle_frame_id]
        with open(crop_metadata_path, "r") as f:
            crop_metadata = json.load(f)
        behavior_frame_id = (muscle_frame_id + 1) * behavior_to_muscle_ratio
        assert crop_metadata["frame_id"] == behavior_frame_id
        shape_in_metadata = tuple(crop_metadata["original_dim"])
        if (
            shape_in_metadata != muscle_image.shape
            and muscle_frame_id == muscle_frame_ids[0]
        ):
            logging.warning(
                f"Muscle image shape {muscle_image.shape} does not match target (behavior) "
                f"shape {shape_in_metadata}. This is likely because the behavior camera "
                f"cannot record at any arbitrary frame size, and it is rounded to the "
                f"nearest allowable size (multiples of 64). Cropping the muscle image "
                f"to the target size."
            )
            assert shape_in_metadata[0] <= muscle_image.shape[0]
            assert shape_in_metadata[1] <= muscle_image.shape[1]
            muscle_image = muscle_image[: shape_in_metadata[0], : shape_in_metadata[1]]
        muscle_image_rotated = rotate_image_around_point(
            image=muscle_image,
            rotation_point=np.array(crop_metadata["rotation"]["original_thorax_pt"]),
            rotation_angle=crop_metadata["rotation"]["angle_radians"],
        )
        muscle_image_cropped, _ = crop_image_and_keypoints(
            image=muscle_image_rotated,
            keypoints=None,
            center=np.array(crop_metadata["crop"]["rotated_thorax_pt"]),
            crop_dim=crop_metadata["crop"]["crop_dim"],
            x_offset=crop_metadata["crop"]["x_offset"],
            y_offset=crop_metadata["crop"]["y_offset"],
        )

        with h5py.File(
            out_dir / f"muscle_segmentation_frame_{muscle_frame_id:05d}.h5", "w"
        ) as f_out:
            f_out.create_dataset(
                "muscle_image",
                data=muscle_image_cropped,
                dtype="uint16",
                compression="gzip",
            )
            f_out.create_dataset(
                "segmap", data=segmap, dtype="uint8", compression="gzip"
            )
            f_out.attrs["class_labels"] = np.array(seg_labels, dtype="S")

        if visualize_masks:
            masks = []
            for seg_value in segs_to_visualize_indices:
                masks.append(segmap == seg_value)
            draw_mask_contours(
                image=muscle_image_cropped,
                masks=np.stack(masks, axis=0),
                vmin=viz_vmin,
                vmax=viz_vmax,
                output_path=viz_out_dir / f"frame_{muscle_frame_id:06d}.png",
            )

    return segmap_by_muscle_frameid


if __name__ == "__main__":
    bodyseg_model_dir = Path("bulk_data/pose_estimation/bodyseg/trial_20251012b/")
    spotlight_basedir = Path("bulk_data/spotlight_recordings/")
    cropped_behavior_basedir = Path(
        "bulk_data/behavior_images/spotlight_aligned_and_cropped/"
    )

    spotlight_trial = "20250613-fly1b-013"
    bodyseg_output_path = (
        bodyseg_model_dir
        / f"inference/{spotlight_trial}_model_prediction_not_flipped/bodyseg_pred.h5"
    )
    spotlight_dir = spotlight_basedir / spotlight_trial
    cropped_behavior_dir = (
        cropped_behavior_basedir / spotlight_trial / "model_prediction/not_flipped"
    )
    original_input_size = (900, 900)

    process_spotlight_trial(
        bodyseg_output_path,
        cropped_behavior_dir,
        spotlight_dir,
        original_input_size=original_input_size,
        visualize_masks=True,
    )
