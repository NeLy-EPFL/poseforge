import h5py
import numpy as np
import yaml
import json
import imageio.v2 as imageio
import logging
import cv2
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm
from joblib import Parallel, delayed

from poseforge.spotlight.input_transform import reverse_rotation_and_crop
from poseforge.spotlight.viz import draw_mask_contours, draw_template_matching_viz


# Define handy constants
leg_segment_names = [
    f"{side}{pos}{link}"
    for side in ["L", "R"]
    for pos in ["F", "M", "H"]
    for link in ["Coxa", "Femur", "Tibia", "Tarsus"]
]
antennal_segment_names = ["LAntenna", "RAntenna"]


def _extract_muscle_trace_single_frame(muscle_image, masks, roi_sizes):
    assert masks.shape[0] == len(roi_sizes)

    muscle_traces = []
    for i in range(masks.shape[0]):
        mask = masks[i, ...]
        assert mask.shape == muscle_image.shape
        selected_pixels = muscle_image[mask]
        # Use ROI area before dilation regardless of whether dilated masks are
        # used. The dilation allows some margin of error for the alignment, but
        # the (projected) size of the physical muscle shouldn't change.
        area = roi_sizes[i]
        # Set muscle activation to NaN if no ROI is detected for this segment
        if area == 0:
            activation = np.nan
        else:
            activation = selected_pixels.mean()
        muscle_traces.append(activation)

    return np.array(muscle_traces)


def process_muscle_segmentation(
    spotlight_trial_dir: Path,
    preprocessed_behavior_image_dir: Path,
    bodyseg_prediction_path: Path,
    output_path: Path,
    muscle_traces_segments: list[str],
    alignment_foreground_segments: list[str] = leg_segment_names,
    muscle_vrange: tuple[int, int] = (200, 1000),
    crop_padding: int = 100,
    template_matching_search_limit: int = 50,
    morph_denoise_kernel_size: int = 5,
    morph_denoise_n_iterations: int = 1,
    dilation_kernel_size: int = 1,
    debug_plots_dir: Path | None = None,
    n_workers: int = -1,
):
    """
    Process muscle segmentation mapping all steps for each frame in parallel.

    Args:
        spotlight_trial_dir (Path):
            Path to spotlight trial directory. This is the "save" directory
            set in the spotlight recording software.
        preprocessed_behavior_image_dir (Path):
            Path to directory containing preprocessed behavior images -
            i.e. after rotating and cropping so that the fly is centered
            and facing up.
        bodyseg_prediction_path (Path):
            Path to H5 file containing predictions from the bodyseg model.
        output_path (Path):
            Path to save muscle segmentation outputs to (in H5 format).
        muscle_traces_segments (list[str]):
            List of body segments to extract fluorescence traces for.
        alignment_foreground_segments (list[str]):
            For template matching, it's useful to only consider certain
            body segments (e.g. legs) as the foreground. This is because
            the muscle images mainly show fluorescence from the legs, and
            other body segments may only introduce noise. This argument
            specifies which body segments to use as foreground for
            template matching. Defaults to all leg segments.
        muscle_vrange (tuple[int, int]):
            Min/max values to normalize muscle images to. This is used for
            template matching and for visualization.
        crop_padding (int):
            For space efficiency, we crop muscle images, body segmentation
            maps, and masks to a bounding box around the non-background
            parts of the segmentation map (with some padding on all sides).
            This argument specifies the padding size in pixels.
            Defaults to 100.
        template_matching_search_limit (int):
            Maximum pixel offset to search in each direction during
            template matching for fine alignment. Defaults to 50.
        morph_denoise_kernel_size (int):
            Kernel size for denoising based on morphological transforms.
            Defaults to 5.
        morph_denoise_n_iterations (int):
            Number of iterations for denoising based on morphological
            transforms. Defaults to 1.
        dilation_kernel_size (int):
            After masks are denoised, we may want to dilate them to provide
            extra margin of error for alignment inaccuracies. This argument
            specifies the dilation kernel size. If set to 1, no dilation is
            performed. Defaults to 1.
        debug_plots_dir (Path | None):
            If provided, debug visualizations for each processing step
            will be saved to this directory. Defaults to None (no plots).
        n_workers (int):
            Number of parallel workers to use (see n_jobs convention in
            joblib). Defaults to -1 (use all available CPU cores).
    """
    # Input validation
    if not spotlight_trial_dir.is_dir():
        raise ValueError(f"Spotlight trial directory not found: {spotlight_trial_dir}")
    if not preprocessed_behavior_image_dir.is_dir():
        raise ValueError(
            f"Preprocessed behavior image directory not found: "
            f"{preprocessed_behavior_image_dir}"
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

    # Get class indices for given lists of segment names
    alignment_foreground_segment_ids = [  # for template matching
        seg_labels.index(label) for label in alignment_foreground_segments
    ]
    muscle_traces_segment_ids = [  # for muscle trace extraction
        seg_labels.index(label) for label in muscle_traces_segments
    ]
    visualized_segment_ids = [  # for drawing ROIs in debug plots
        seg_labels.index(label) for label in leg_segment_names + antennal_segment_names
    ]

    input_kwargs_all_tasks = []
    for muscle_frame_id, muscle_path in muscle_path_by_frameid.items():
        behavior_frame_id = (muscle_frame_id + 1) * sync_ratio
        if behavior_frame_id not in behavior_frame_to_stack_idx:
            continue

        segmap_idx = behavior_frame_to_stack_idx[behavior_frame_id]
        segmap = pred_segmaps[segmap_idx]

        metadata_path = (
            preprocessed_behavior_image_dir.parent.parent
            / "all"
            / f"frame_{behavior_frame_id:09d}.metadata.json"
        )

        input_kwargs = {
            "muscle_frame_id": muscle_frame_id,
            "muscle_path": muscle_path,
            "segmap": segmap,
            "metadata_path": metadata_path,
            "padding": crop_padding,
            "muscle_traces_segment_ids": muscle_traces_segment_ids,
            "alignment_foreground_segment_ids": alignment_foreground_segment_ids,
            "search_limit": template_matching_search_limit,
            "morph_kernel_size": morph_denoise_kernel_size,
            "morph_n_iterations": morph_denoise_n_iterations,
            "dilation_kernel_size": dilation_kernel_size,
            "muscle_vrange": muscle_vrange,
            "seg_labels": seg_labels,
            "segs_to_visualize_ids": visualized_segment_ids,
            "debug_plots_dir": debug_plots_dir,
        }
        input_kwargs_all_tasks.append(input_kwargs)

    # Setup debug visualization directory
    if debug_plots_dir is not None:
        debug_plots_dir.mkdir(parents=True, exist_ok=True)

    # Process all frames in parallel
    parallel_processor = Parallel(n_jobs=n_workers, verbose=1)
    logging.info(
        f"Processing {len(input_kwargs_all_tasks)} frames "
        f"with {parallel_processor._effective_n_jobs} workers"
    )
    results_all_tasks = parallel_processor(
        delayed(_process_single_frame)(**input_kwargs)
        for input_kwargs in tqdm(
            input_kwargs_all_tasks, disable=None, desc="Processing frames"
        )
    )

    # Save results to H5 file
    logging.info(f"Saving results to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        for input_kwargs, (result_datasets, result_attributes) in tqdm(
            zip(input_kwargs_all_tasks, results_all_tasks),
            disable=None,
            total=len(results_all_tasks),
            desc="Saving results",
        ):
            muscle_frame_id = input_kwargs["muscle_frame_id"]
            grp = f.create_group(f"muscle_frame_{muscle_frame_id:06d}")
            for key, data in result_datasets.items():
                grp.create_dataset(key, data=data, compression="gzip", shuffle=True)
            grp.attrs.update(result_attributes)
            grp.attrs["all_segments"] = seg_labels
            grp.attrs["alignment_foreground_segments"] = alignment_foreground_segments
            grp.attrs["muscle_traces_segments"] = muscle_traces_segments

    logging.info(f"Saved results to {output_path}")


def _load_and_map_segmentation_to_muscle_space(
    muscle_path: Path, segmap: np.ndarray, metadata_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Load muscle image & map segmentation from behavior to muscle space.
    
    Args:
        See `process_muscle_segmentation`.
        
    Returns:
        muscle_image (np.ndarray):
            Loaded muscle image, cropped to where segmentation map
            indicates non-background features are.
        segmap_mapped (np.ndarray):
            Segmentation map transformed to muscle image space, also
            cropped to match the muscle image.
    """
    # Load muscle image and metadata
    muscle_image = imageio.imread(muscle_path)
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Handle size mismatch between muscle and behavior images
    size_before_transform = tuple(metadata["rotation"]["input_size"])
    if size_before_transform != muscle_image.shape:
        # The muscle image and the behavior image (whose size is specified by
        # size_before_transform) can be slightly different. This is because the behavior
        # camera cannot record at any image size. The image size is rounded to the
        # nearest allowed value (multiple of 64). If this happens, we crop the two
        # images to the same size.
        # ! ONLY ONE CASE IS HANDLED HERE - THIS ISSUE WILL BE FIXED IN SPOTLIGHT
        # ! CONTROL SOFTWARE AND THIS CODEPATH WILL BE REMOVED IN THE FUTURE.
        assert size_before_transform[0] == muscle_image.shape[0]
        assert size_before_transform[1] == 1472
        assert muscle_image.shape[1] == 1500
        muscle_image = muscle_image[
            : size_before_transform[0], : size_before_transform[1]
        ]

    # Resize segmap from model working resolution back to preprocessed image size
    # The behavior image preprocessing step (rotation + crop) does NOT change the size
    # of the image. The preprocessed image can still be rather large (e.g. 900x900).
    # However, the pose models (including bodyseg) first resize the input to a smaller
    # working dimension (e.g. 256x256). The output segmap is therefore at this smaller
    # size. Before reversing the preprocessing transforms, we need to first resize the
    # segmap back to the preprocessed image size.
    size_after_crop = metadata["crop"]["output_size"]
    zoom_factor = (
        size_after_crop[0] / segmap.shape[0],
        size_after_crop[1] / segmap.shape[1],
    )
    segmap_resized = ndimage.zoom(segmap, zoom_factor, order=0)

    # Reverse the preprocessing transforms (rotation + crop) applied to behavior images
    segmap_mapped = reverse_rotation_and_crop(
        segmap_resized,
        rotation_params=metadata["rotation"],
        crop_params=metadata["crop"],
        fill_value=0,
    )
    
    return muscle_image, segmap_mapped


def _process_single_frame(
    muscle_frame_id: int,
    muscle_path: Path,
    segmap: np.ndarray,
    metadata_path: Path,
    padding: int,
    muscle_traces_segment_ids: list[int],
    alignment_foreground_segment_ids: list[int],
    search_limit: int,
    morph_kernel_size: int,
    morph_n_iterations: int,
    dilation_kernel_size: int,
    muscle_vrange: tuple[int, int],
    seg_labels: list[str],
    segs_to_visualize_ids: list[int] | None,
    debug_plots_dir: Path | None,
) -> tuple[dict, dict]:
    """Process all segmentation steps for a single frame.

    Args:
        See `process_muscle_segmentation`.

    Returns:
        datasets (dict[str, np.ndarray]):
            Contains cropped images, aligned segmaps, masks, etc.
        attributes (dict[str, any]):
            Contains metadata like shifts, pixel counts, file paths, etc.
    """

    # Step 1: Load and map segmentation to muscle space
    muscle_image, segmap_mapped = _load_and_map_segmentation_to_muscle_space(
        muscle_path, segmap, metadata_path
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
    foreground_mask = np.isin(segmap_cropped, alignment_foreground_segment_ids)

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
        segmap_aligned, seg_labels, morph_kernel, morph_n_iterations
    )

    # Step 4: Dilate masks to provide extra margin for alignment errors
    if dilation_kernel_size == 1:
        final_masks = denoised_masks
    else:
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size)
        )
        final_masks = _dilate_masks(denoised_masks, seg_labels, dilation_kernel)

    # Step 5: Extract muscle traces for requested segments
    muscle_masks = final_masks[muscle_traces_segment_ids, ...]
    # Use ROI area before dilation regardless of whether dilated masks are used. The
    # dilation allows some margin of error for the alignment, but the (projected) size
    # of the physical muscle shouldn't change.
    muscle_roi_sizes = [denoised_masks[i].sum() for i in muscle_traces_segment_ids]
    muscle_traces = _extract_muscle_trace_single_frame(
        muscle_cropped, muscle_masks, muscle_roi_sizes
    )

    # Debug visualizations for each step
    if debug_plots_dir is not None and segs_to_visualize_ids is not None:
        # Step 1: Initial mapping visualization
        initial_mapping_dir = debug_plots_dir / "01_initial_mapping"
        initial_mapping_dir.mkdir(parents=True, exist_ok=True)
        masks = []
        for seg_value in segs_to_visualize_ids:
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
        masks_for_viz = denoised_masks[segs_to_visualize_ids]
        draw_mask_contours(
            image=muscle_cropped,
            masks=masks_for_viz,
            muscle_vrange=muscle_vrange,
            output_path=morph_denoising_dir / f"frame_{muscle_frame_id:06d}.jpg",
        )

        # Step 4: Dilation visualization (if applied)
        if dilation_kernel_size != 1:
            dilation_dir = debug_plots_dir / "04_mask_dilation"
            dilation_dir.mkdir(parents=True, exist_ok=True)
            masks_for_viz = final_masks[segs_to_visualize_ids]
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
        "muscle_traces": muscle_traces,
    }

    if dilation_kernel_size != 1:
        datasets["masks_dilated"] = final_masks.astype(np.bool_)

    attributes = {
        "muscle_image_path": str(muscle_path.absolute()),
        "input_transform_metadata_path": str(metadata_path.absolute()),
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
        has_feature (bool):
            Whether any non-background features were found.
        muscle_cropped (np.ndarray):
            Cropped muscle image, shape (h, w)
        segmap_cropped (np.ndarray):
            Cropped segmentation map, shape (h, w)
        crop_metadata (dict):
            Contains bounding box information:
            - segmap_feature_bbox_row_range: [min_row, max_row]
            - segmap_feature_bbox_col_range: [min_col, max_col]
            - crop_box_row_range: [crop_min_row, crop_max_row]
            - crop_box_col_range: [crop_min_col, crop_max_col]
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
        x_shift: int
            Optimal column-wise translation to be applied to the segmap.
        y_shift: int
            Optimal row-wise translation to be applied to the segmap.
        correlation_matrix: np.ndarray
            Cross-correlation matrix of the template matching.
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
        Array of denoised binary masks, shape (len(class_labels), H, W)
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
        class_labels: List of class names
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
