import h5py
import numpy as np
import yaml
import json
import imageio.v2 as imageio
import logging
import cv2
from pathlib import Path
from collections import defaultdict
from scipy import ndimage
from tqdm import tqdm
from joblib import Parallel, delayed

from poseforge.spotlight.input_transform import reverse_rotation_and_crop
from poseforge.neuromechfly.constants import legs
from poseforge.spotlight.viz import draw_mask_contours, draw_template_matching_viz


# Define handy constants
leg_segment_names = [
    f"{side}{pos}{link}"
    for side in ["L", "R"]
    for pos in ["F", "M", "H"]
    for link in ["Coxa", "Femur", "Tibia", "Tarsus"]
]
antennal_segment_names = ["LAntenna", "RAntenna"]


def extract_muscle_trace(
    muscle_segmentation_path: Path, body_segments: list[str], use_dilated: bool = True
):
    with h5py.File(muscle_segmentation_path, "r") as f:
        # Get indices of requested body segments in the stack of masks
        class_labels = list(f.attrs["class_labels"])
        segment_stack_indices = []
        for segment in body_segments:
            if segment not in class_labels:
                logging.critical(
                    f"Body segment '{segment}' not found in segmentation labels: "
                    f"{class_labels}"
                )
            segment_stack_indices.append(class_labels.index(segment))
        if len(segment_stack_indices) != len(body_segments):
            raise ValueError("Some body segments not found in segmentation labels.")

        # Check which processed step to use depending on the use_dilated flag
        if use_dilated:
            if "masks_dilated" not in f[list(f.keys())[0]]:
                raise ValueError(
                    "Dilated masks not found in muscle segmentation file. "
                    "Set use_dilated=False or re-run muscle segmentation with "
                    "dilation enabled."
                )
            dataset_name = "masks_dilated"
        else:
            dataset_name = "masks_morph_denoised"

        # Extract muscle traces for each requested body segment
        muscle_traces = defaultdict(dict)
        muscle_pixel_values = defaultdict(dict)
        empty_mask_counter = defaultdict(int)
        for frame_key in sorted(f.keys()):
            muscle_frame_id = int(frame_key.split("_")[-1])
            grp = f[frame_key]
            muscle_image = grp["muscle_image_cropped"][:]
            for idx, label in zip(segment_stack_indices, body_segments):
                mask = grp[dataset_name][idx, ...]
                assert mask.shape == muscle_image.shape
                selected_pixels = muscle_image[mask]
                # Use ROI area before dilation regardless of whether dilated masks are
                # used. The dilation allows some margin of error for the alignment, but
                # the (projected) size of the physical muscle shouldn't change.
                area = grp.attrs[f"n_pixels_pre_dilation_{label}"]
                # Set muscle activation to NaN if no ROI is detected for this segment
                if area == 0:
                    activation = np.nan
                    empty_mask_counter[label] += 1
                else:
                    activation = selected_pixels.mean()
                muscle_traces[label][muscle_frame_id] = activation
                muscle_pixel_values[label][muscle_frame_id] = selected_pixels.copy()

        # Log summary of empty masks
        n_frames = len(f.keys())
        for label, count in empty_mask_counter.items():
            if count > 0:
                logging.warning(
                    f"Segment '{label}' had {count} out of {n_frames} frames with "
                    "empty masks. Muscle activations set to NaN for these frames."
                )

    return muscle_traces, muscle_pixel_values


def process_muscle_segmentation(
    spotlight_trial_dir: Path,
    aligned_behavior_image_dir: Path,
    bodyseg_prediction_path: Path,
    output_path: Path,
    foreground_classes_for_alignment: list[str],
    muscle_vrange: tuple[int, int] = (200, 1000),
    padding: int = 100,
    search_limit: int = 50,
    morph_kernel_size: int = 5,
    morph_iterations: int = 1,
    dilation_size: int | None = None,
    debug_plots_dir: Path | None = None,
    n_workers: int = -1,
):
    """
    Process muscle segmentation mapping all steps for each frame in parallel.

    Args:
        spotlight_trial_dir: Path to spotlight trial directory
        aligned_behavior_image_dir: Path to aligned behavior images
        bodyseg_prediction_path: Path to bodyseg predictions H5 file
        output_path: Output H5 file path
        muscle_vrange: Min/max values for muscle image visualization
        padding: Padding around bounding box for cropping
        foreground_classes_for_alignment: Classes to use for template
            matching foreground
        search_limit: Max pixel offset for template matching
        morph_kernel_size: Kernel size for morphological operations
        morph_iterations: Number of morphological iterations
        dilation_size: Dilation kernel size (None to skip dilation)
        debug_plots_dir: Optional directory for debug visualizations
        n_workers: Number of parallel workers
    """
    logging.info("Processing muscle segmentation mapping")

    # Input validation
    if not spotlight_trial_dir.is_dir():
        raise ValueError(f"Spotlight trial directory not found: {spotlight_trial_dir}")
    if not aligned_behavior_image_dir.is_dir():
        raise ValueError(
            f"Aligned behavior image directory not found: {aligned_behavior_image_dir}"
        )
    if not bodyseg_prediction_path.is_file():
        raise ValueError(
            f"Bodyseg prediction file not found: {bodyseg_prediction_path}"
        )

    # Load bodyseg predictions
    logging.info(f"Loading bodyseg output from: {bodyseg_prediction_path}")
    with h5py.File(bodyseg_prediction_path, "r") as f:
        behavior_frame_ids = list(f["frame_ids"][:])
        pred_segmaps = f["pred_segmap"][:]
        seg_labels = list(f["pred_segmap"].attrs["class_labels"])

    # Get muscle image paths
    muscle_image_paths = list(
        spotlight_trial_dir.glob("processed/muscle_images/muscle_frame_*.tif")
    )
    muscle_path_by_frameid = {
        int(path.stem.split("_")[-1]): path for path in sorted(muscle_image_paths)
    }

    # Load sync ratio
    dual_recording_metadata_path = (
        spotlight_trial_dir / "metadata/dual_recording_timing.yaml"
    )
    with open(dual_recording_metadata_path, "r") as f:
        sync_ratio = yaml.safe_load(f)["sync_ratio"]
    if int(sync_ratio) != sync_ratio:
        raise ValueError(
            f"Sync ratio must be an integer, got {sync_ratio} from "
            f"{dual_recording_metadata_path}"
        )
    sync_ratio = int(sync_ratio)

    # Match muscle frames to behavior frames
    behavior_frame_to_stack_idx = {
        frame_id: idx for idx, frame_id in enumerate(behavior_frame_ids)
    }

    frame_tasks = []
    for muscle_frame_id, muscle_path in muscle_path_by_frameid.items():
        behavior_frame_id = (muscle_frame_id + 1) * sync_ratio
        if behavior_frame_id not in behavior_frame_to_stack_idx:
            continue

        segmap_idx = behavior_frame_to_stack_idx[behavior_frame_id]
        segmap = pred_segmaps[segmap_idx]

        metadata_path = (
            aligned_behavior_image_dir.parent.parent
            / "all"
            / f"frame_{behavior_frame_id:09d}.metadata.json"
        )

        frame_tasks.append(
            {
                "muscle_frame_id": muscle_frame_id,
                "muscle_path": muscle_path,
                "segmap": segmap,
                "metadata_path": metadata_path,
            }
        )

    logging.info(f"Processing {len(frame_tasks)} frames with {n_workers} workers")

    # Get foreground class indices for template matching
    foreground_class_indices = [
        seg_labels.index(label) for label in foreground_classes_for_alignment
    ]

    # Setup debug visualization
    segs_to_visualize_indices = None
    if debug_plots_dir is not None:
        debug_plots_dir.mkdir(parents=True, exist_ok=True)
        segs_to_visualize_indices = [
            seg_labels.index(label)
            for label in leg_segment_names + antennal_segment_names
        ]

    # Process all frames in parallel
    results = Parallel(n_jobs=n_workers, verbose=1)(
        delayed(_process_single_frame)(
            task=task,
            padding=padding,
            foreground_class_indices=foreground_class_indices,
            search_limit=search_limit,
            morph_kernel_size=morph_kernel_size,
            morph_iterations=morph_iterations,
            dilation_size=dilation_size,
            muscle_vrange=muscle_vrange,
            seg_labels=seg_labels,
            segs_to_visualize_indices=segs_to_visualize_indices,
            debug_plots_dir=debug_plots_dir,
        )
        for task in tqdm(frame_tasks, desc="Processing frames")
    )

    # Save results to H5 file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["class_labels"] = seg_labels

        for task, (result_datasets, result_attributes) in zip(frame_tasks, results):
            if result_datasets is None:  # Frame failed processing
                continue

            grp = f.create_group(f"muscle_frame_{task['muscle_frame_id']:06d}")

            # Save all datasets
            for key, data in result_datasets.items():
                grp.create_dataset(key, data=data, compression="gzip", shuffle=True)
            grp.attrs.update(result_attributes)

    logging.info(f"Saved results to {output_path}")


def _process_single_frame(
    task: dict,
    padding: int,
    foreground_class_indices: list[int],
    search_limit: int,
    morph_kernel_size: int,
    morph_iterations: int,
    dilation_size: int | None,
    muscle_vrange: tuple[int, int],
    seg_labels: list[str],
    segs_to_visualize_indices: list[int] | None,
    debug_plots_dir: Path | None,
) -> tuple[dict, dict]:
    """Process all segmentation steps for a single frame.

    Args:
        See `process_muscle_segmentation`.

    Returns:
        Tuple of (datasets_dict, attributes_dict) where:
        - datasets_dict: Contains cropped images, aligned segmaps, masks, etc.
        - attributes_dict: Contains metadata like shifts, pixel counts, file paths
    """

    muscle_frame_id = task["muscle_frame_id"]

    # Step 1: Load and map segmentation to muscle space
    muscle_image = imageio.imread(task["muscle_path"])
    segmap = task["segmap"]

    with open(task["metadata_path"], "r") as f:
        metadata = json.load(f)

    # Resize and reverse transform segmap to muscle space
    size_before_transform = tuple(metadata["rotation"]["input_size"])
    if size_before_transform != muscle_image.shape:
        muscle_image = muscle_image[
            : size_before_transform[0], : size_before_transform[1]
        ]

    size_after_crop = metadata["crop"]["output_size"]
    zoom_factor = (
        size_after_crop[0] / segmap.shape[0],
        size_after_crop[1] / segmap.shape[1],
    )
    segmap_resized = ndimage.zoom(segmap, zoom_factor, order=0)

    segmap_mapped = reverse_rotation_and_crop(
        segmap_resized,
        rotation_params=metadata["rotation"],
        crop_params=metadata["crop"],
        fill_value=0,
    )

    # Crop to bounding box
    has_feature, muscle_cropped, segmap_cropped, crop_metadata = _crop_by_features(
        muscle_image, segmap_mapped, padding
    )

    if not has_feature:
        logging.warning(
            f"No non-background features found in muscle frame {muscle_frame_id}. "
            "This is extremely unlikely. Check upstream processing."
        )
        # Continue processing - all masks will be empty which is fine

    # Step 2: Fine alignment via template matching
    muscle_norm = _normalize_muscle_image(muscle_cropped, muscle_vrange)
    foreground_mask = np.isin(segmap_cropped, foreground_class_indices)

    x_shift, y_shift, corr_matrix = _template_match(
        muscle_norm, foreground_mask, search_limit
    )
    segmap_aligned = ndimage.shift(
        segmap_cropped, (y_shift, x_shift), order=0, mode="nearest"
    )

    # Step 3: Morphological denoising
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    denoised_masks = _denoise_masks(
        segmap_aligned, seg_labels, morph_kernel, morph_iterations
    )

    # Step 4: Optional dilation
    final_masks = denoised_masks
    if dilation_size is not None:
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_size, dilation_size)
        )
        final_masks = _dilate_masks(denoised_masks, seg_labels, dilation_kernel)

    # Debug visualizations for each step
    if debug_plots_dir is not None and segs_to_visualize_indices is not None:
        # Step 1: Initial mapping visualization
        initial_mapping_dir = debug_plots_dir / "01_initial_mapping"
        initial_mapping_dir.mkdir(parents=True, exist_ok=True)
        masks = []
        for seg_value in segs_to_visualize_indices:
            masks.append(segmap_cropped == seg_value)
        draw_mask_contours(
            image=muscle_cropped,
            masks=np.stack(masks, axis=0),
            muscle_vrange=muscle_vrange,
            output_path=initial_mapping_dir / f"frame_{muscle_frame_id:06d}.jpg",
        )

        # Step 2: Fine alignment visualization
        fine_alignment_dir = debug_plots_dir / "02_fine_alignment"
        fine_alignment_dir.mkdir(parents=True, exist_ok=True)
        draw_template_matching_viz(
            muscle_image=muscle_cropped,
            foreground_mask=foreground_mask,
            x_shift=x_shift,
            y_shift=y_shift,
            corr_matrix=corr_matrix,
            search_limit=search_limit,
            output_path=fine_alignment_dir / f"frame_{muscle_frame_id:06d}.jpg",
            muscle_vrange=muscle_vrange,
        )

        # Step 3: Morphological denoising visualization
        morph_denoising_dir = debug_plots_dir / "03_morph_denoising"
        morph_denoising_dir.mkdir(parents=True, exist_ok=True)
        masks_for_viz = denoised_masks[segs_to_visualize_indices]
        draw_mask_contours(
            image=muscle_cropped,
            masks=masks_for_viz,
            muscle_vrange=muscle_vrange,
            output_path=morph_denoising_dir / f"frame_{muscle_frame_id:06d}.jpg",
        )

        # Step 4: Dilation visualization (if applied)
        if dilation_size is not None:
            dilation_dir = debug_plots_dir / "04_mask_dilation"
            dilation_dir.mkdir(parents=True, exist_ok=True)
            masks_for_viz = final_masks[segs_to_visualize_indices]
            draw_mask_contours(
                image=muscle_cropped,
                masks=masks_for_viz,
                muscle_vrange=muscle_vrange,
                output_path=dilation_dir / f"frame_{muscle_frame_id:06d}.jpg",
            )

    # Prepare output
    datasets = {
        "muscle_image_cropped": muscle_cropped,
        "segmap_cropped": segmap_cropped,
        "segmap_fine_aligned": segmap_aligned,
        "template_matching_correlation_matrix": corr_matrix,
        "masks_morph_denoised": denoised_masks.astype(np.bool_),
    }

    if dilation_size is not None:
        datasets["masks_dilated"] = final_masks.astype(np.bool_)

    attributes = {
        "muscle_image_path": str(task["muscle_path"].absolute()),
        "input_transform_metadata_path": str(task["metadata_path"].absolute()),
        "has_non_background_feature": has_feature,
        "template_matching_x_shift": x_shift,
        "template_matching_y_shift": y_shift,
        **crop_metadata,
    }

    # Add pixel counts
    for i, label in enumerate(seg_labels):
        attributes[f"n_pixels_pre_dilation_{label}"] = int(denoised_masks[i].sum())

    return datasets, attributes


def _crop_by_features(
    muscle_image: np.ndarray, segmap: np.ndarray, padding: int
) -> tuple[bool, np.ndarray, np.ndarray, dict[str, list[int]]]:
    """Crop images to bounding box of non-background features.

    Args:
        muscle_image: Muscle image array, shape (H, W)
        segmap: Segmentation map array, shape (H, W)
        padding: Number of pixels to pad around the bounding box

    Returns:
        Tuple of (has_feature, muscle_cropped, segmap_cropped, crop_metadata)
        where crop_metadata contains bounding box information.
    """
    non_background_mask = segmap > 0

    if non_background_mask.sum() == 0:
        has_feature = False
        return has_feature, muscle_image, segmap, {}

    ys, xs = np.where(non_background_mask)
    feature_row_min, feature_row_max = ys.min(), ys.max()
    feature_col_min, feature_col_max = xs.min(), xs.max()

    crop_row_min = max(0, feature_row_min - padding)
    crop_row_max = min(muscle_image.shape[0], feature_row_max + padding)
    crop_col_min = max(0, feature_col_min - padding)
    crop_col_max = min(muscle_image.shape[1], feature_col_max + padding)

    muscle_cropped = muscle_image[crop_row_min:crop_row_max, crop_col_min:crop_col_max]
    segmap_cropped = segmap[crop_row_min:crop_row_max, crop_col_min:crop_col_max]

    metadata = {
        "segmap_feature_bbox_row_range": [feature_row_min, feature_row_max],
        "segmap_feature_bbox_col_range": [feature_col_min, feature_col_max],
        "crop_box_row_range": [crop_row_min, crop_row_max],
        "crop_box_col_range": [crop_col_min, crop_col_max],
    }

    has_feature = True
    return has_feature, muscle_cropped, segmap_cropped, metadata


def _normalize_muscle_image(
    muscle_image: np.ndarray, muscle_vrange: tuple[int, int]
) -> np.ndarray:
    """Normalize muscle image to 0-1 range."""
    vmin, vmax = muscle_vrange
    return np.clip((muscle_image.astype(np.float32) - vmin) / (vmax - vmin), 0, 1)


def _template_match(
    muscle_image: np.ndarray, mask: np.ndarray, search_limit: int
) -> tuple[int, int, np.ndarray]:
    """Find optimal translation using template matching.

    Args:
        muscle_image: Normalized muscle image, shape (H, W)
        mask: Binary foreground mask, shape (H, W)
        search_limit: Maximum pixel offset to search in any direction

    Returns:
        Tuple of (x_shift, y_shift, correlation_matrix)
    """
    n_rows, n_cols = muscle_image.shape

    # Pad muscle image
    padded_image = np.zeros(
        (n_rows + 2 * search_limit, n_cols + 2 * search_limit), dtype=np.float32
    )
    padded_image[
        search_limit : search_limit + n_rows, search_limit : search_limit + n_cols
    ] = muscle_image

    # Template matching
    corr_matrix = cv2.matchTemplate(
        padded_image, mask.astype(np.float32), cv2.TM_CCORR_NORMED
    )
    _, _, _, max_loc = cv2.minMaxLoc(corr_matrix)

    x_shift = max_loc[0] - search_limit
    y_shift = max_loc[1] - search_limit

    return x_shift, y_shift, corr_matrix


def _denoise_masks(
    segmap: np.ndarray,
    class_labels: list[str],
    morph_kernel: np.ndarray,
    n_iterations: int,
) -> np.ndarray:
    """Apply morphological denoising to all class masks.

    Args:
        segmap: Segmentation map with class labels, shape (H, W)
        class_labels: List of class names corresponding to segmap values
        morph_kernel: Structuring element for morphological operations.
        n_iterations: Number of iterations for opening and closing

    Returns:
        Array of denoised binary masks, shape (n_classes, H, W)
    """
    denoised_masks = []

    for i, class_label in enumerate(class_labels):
        mask = (segmap == i).astype(np.uint8)

        if class_label.lower() == "background":
            denoised_mask = mask.astype(bool)
        else:
            # Morphological opening and closing
            opened = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, morph_kernel, iterations=n_iterations
            )
            closed = cv2.morphologyEx(
                opened, cv2.MORPH_CLOSE, morph_kernel, iterations=n_iterations
            )

            # Keep largest connected component
            num_labels, labels_im = cv2.connectedComponents(closed)
            if num_labels <= 1:
                denoised_mask = np.zeros_like(mask, dtype=bool)
            else:
                largest_component = 1 + np.argmax(
                    [np.sum(labels_im == label_id) for label_id in range(1, num_labels)]
                )
                denoised_mask = labels_im == largest_component

        denoised_masks.append(denoised_mask)

    return np.stack(denoised_masks, axis=0)


def _dilate_masks(
    masks: np.ndarray, class_labels: list[str], dilation_kernel: np.ndarray
) -> np.ndarray:
    """Apply dilation to masks (skip background).

    Args:
        masks: Binary masks to dilate, shape (n_classes, H, W)
        class_labels: List of class names (background class is skipped)
        dilation_kernel: Structuring element for dilation

    Returns:
        Array of dilated binary masks, shape (n_classes, H, W)
    """
    dilated_masks = []

    for i, class_label in enumerate(class_labels):
        mask = masks[i].astype(np.uint8)

        if class_label.lower() == "background":
            dilated_mask = mask.astype(bool)
        else:
            dilated_mask = cv2.dilate(mask, dilation_kernel, iterations=1).astype(bool)

        dilated_masks.append(dilated_mask)

    return np.stack(dilated_masks, axis=0)
