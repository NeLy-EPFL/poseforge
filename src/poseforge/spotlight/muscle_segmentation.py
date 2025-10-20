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
from poseforge.neuromechfly.constants import legs
from poseforge.spotlight.viz import draw_mask_contours, draw_template_matching_viz


class MuscleSegmentationPipeline:
    def __init__(
        self,
        spotlight_trial_dir: Path,
        aligned_behavior_image_dir: Path,
        bodyseg_prediction_path: Path,
        output_path: Path,
        muscle_vrange: tuple[int, int] = (200, 1000),
        debug_plots_dir: Path | None = None,
    ):
        self.spotlight_trial_dir = spotlight_trial_dir
        self.cropped_behavior_image_dir = aligned_behavior_image_dir
        self.bodyseg_prediction_path = bodyseg_prediction_path
        self.output_path = output_path
        self.muscle_vrange = muscle_vrange
        self.debug_plots_dir = debug_plots_dir

        if not spotlight_trial_dir.is_dir():
            raise ValueError(f"Spotlight trial dir not found: {spotlight_trial_dir}")
        if not aligned_behavior_image_dir.is_dir():
            raise ValueError(
                f"Cropped behavior image dir not found: {aligned_behavior_image_dir}"
            )
        if not bodyseg_prediction_path.is_file():
            raise ValueError(
                f"Bodyseg prediction path not found: {bodyseg_prediction_path}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)

    def apply_initial_mapping(self, padding: int = 100, n_workers: int = -1):
        """
        Map bodyseg segmentation masks to muscle images.

        This method processes spotlight trial data by:
        1. Loading muscle images and bodyseg predictions
        2. Matching muscle frames to behavior frames using sync ratio
        3. Applying reverse transformations to map segmaps to muscle space
        4. Cropping around detected features with padding
        5. Saving results to HDF5 file

        Args:
            padding: Padding around the bounding box for cropping.
            n_workers: Number of parallel workers for processing frames.
                Use 1 for sequential processing, -1 for all available cores.
        """
        logging.info("Mapping bodyseg segmentation maps to muscle images")

        muscle_path_by_muscle_frameid = self._collect_muscle_image_paths()

        # Load bodyseg masks and match to muscle frames
        behavior_frame_ids, pred_segmaps, seg_labels = _load_bodyseg_predictions(
            self.bodyseg_prediction_path
        )

        # Find corresponding segmaps (i.e. behavior frame) for each muscle frame
        segmap_stack_idx_by_muscle_frameid, metadata_path_by_muscle_frameid = (
            self._match_muscle_frames_to_behavior(
                muscle_path_by_muscle_frameid, behavior_frame_ids
            )
        )
        segmap_by_muscle_frameid = {
            muscle_frame_id: pred_segmaps[stack_idx, :, :]
            for muscle_frame_id, stack_idx in segmap_stack_idx_by_muscle_frameid.items()
        }

        # Treat each frame as an independent task. Prepare inputs for all frames and
        # process in parallel.
        input_kwargs_all_frames = self._prepare_initial_mapping_inputs(
            segmap_by_muscle_frameid,
            muscle_path_by_muscle_frameid,
            metadata_path_by_muscle_frameid,
            padding,
            seg_labels,
        )
        results_all_frames = Parallel(n_jobs=n_workers, verbose=1)(
            delayed(_map_segmap_to_muscle_single_frame)(**kwargs)
            for kwargs in tqdm(
                input_kwargs_all_frames, disable=None, desc="Mapping segmap"
            )
        )

        # Save mapped segmaps and cropped muscle images to H5 file
        muscle_frame_ids = list(segmap_by_muscle_frameid.keys())
        with h5py.File(self.output_path, "w") as f_out:
            self._save_initial_mapping_results(
                f_out,
                muscle_frame_ids,
                input_kwargs_all_frames,
                results_all_frames,
                seg_labels,
            )

    def apply_fine_alignment(
        self,
        foreground_classes: list[str] | str = "legs",
        search_limit: int = 50,
        n_workers=-1,
    ):
        """
        Apply fine alignment using template matching to improve segmentation accuracy.

        This method addresses imperfections in the initial calibration-based alignment
        by using normalized cross-correlation to find optimal translations between
        foreground masks and muscle images.

        Args:
            foreground_classes: List of class names to use as foreground for
                template matching. If "legs", uses all leg segments.
            search_limit: Maximum offset to search in any direction (pixels).
            n_workers: Number of parallel workers for processing frames.
        """
        plots_subdir = None
        if self.debug_plots_dir is not None:
            plots_subdir = self.debug_plots_dir / "template_matching"
            plots_subdir.mkdir(parents=True, exist_ok=True)

        # Load class labels
        with h5py.File(self.output_path, "r") as f:
            class_labels = list(f.attrs["class_labels"])
        foreground_class_indices = _get_foreground_class_ids(
            foreground_classes, class_labels
        )

        # Prepare inputs for all frames
        input_kwargs_all_frames = self._prepare_fine_alignment_inputs(
            search_limit, foreground_class_indices, self.muscle_vrange, plots_subdir
        )

        # Process all frames in parallel
        results_all_frames = Parallel(n_jobs=n_workers, verbose=1)(
            delayed(_template_match_mask_to_muscle_single_frame)(**kwargs)
            for kwargs in tqdm(
                input_kwargs_all_frames,
                disable=None,
                desc="Fine alignment",
            )
        )

        with h5py.File(self.output_path, "a") as f:
            for frame_key, result in zip(f.keys(), results_all_frames):
                grp = f[frame_key]
                grp.create_dataset(
                    "segmap_fine_aligned",
                    data=result["segmap_fine_aligned"],
                    compression="gzip",
                    shuffle=True,
                )
                grp.create_dataset(
                    "template_matching_correlation_matrix",
                    data=result["corr_matrix"],
                    compression="gzip",
                )
                grp.attrs["template_matching_x_shift"] = result["x_shift"]
                grp.attrs["template_matching_y_shift"] = result["y_shift"]

    def apply_morph_denoising(
        self,
        kernel_size: int = 5,
        n_iterations: int = 2,
        n_workers: int = -1,
    ):
        """
        Apply morphological denoising to fine-aligned segmentation maps.

        This method uses morphological opening and closing to remove small
        artifacts and smooth the segmentation masks. Then, it keeps only
        the largest connected component for each class label.

        Args:
            kernel_size: Size of the structuring element for morphological
                operations.
            n_iterations: Number of iterations for opening and closing.
            n_workers: Number of parallel workers for processing frames.
        """
        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        if self.debug_plots_dir is None:
            viz_output_dir = None
        else:
            viz_output_dir = self.debug_plots_dir / "morph_denoising"
            viz_output_dir.mkdir(parents=True, exist_ok=True)
        input_kwargs_all_frames = self._prepare_morph_denoising_inputs(
            morph_kernel, n_iterations, viz_output_dir=viz_output_dir
        )
        results_all_frames = Parallel(n_jobs=n_workers, verbose=1)(
            delayed(_denoise_mask_for_each_class_single_frame)(**kwargs)
            for kwargs in tqdm(
                input_kwargs_all_frames, disable=None, desc="Morph. denoising"
            )
        )
        with h5py.File(self.output_path, "a") as f:
            class_labels = list(f.attrs["class_labels"])
            for frame_key, results in zip(f.keys(), results_all_frames):
                grp = f[frame_key]
                denoised_masks = results["denoised_masks"]
                ds = grp.create_dataset(
                    "masks_morph_denoised",
                    data=denoised_masks.astype(np.bool_),
                    compression="gzip",
                    shuffle=True,
                )
                n_pixels_per_class = denoised_masks.sum(axis=(1, 2))
                for label, n_pixels in zip(class_labels, n_pixels_per_class):
                    ds.attrs[f"n_pixels_{label}"] = n_pixels

    def _collect_muscle_image_paths(self) -> dict[int, Path]:
        """Collect muscle image file paths indexed by frame ID.

        Returns:
            Dictionary mapping muscle frame IDs to image file paths.
        """
        muscle_image_paths = list(
            self.spotlight_trial_dir.glob("processed/muscle_images/muscle_frame_*.tif")
        )
        return {
            int(path.stem.split("_")[-1]): path for path in sorted(muscle_image_paths)
        }

    def _get_behavior_to_muscle_ratio(self) -> float:
        """Load behavior-to-muscle frame sync ratio from metadata.

        Returns:
            Sync ratio for converting between muscle and behavior frame IDs.
        """
        dual_recording_metadata_path = (
            self.spotlight_trial_dir / "metadata/dual_recording_timing.yaml"
        )
        with open(dual_recording_metadata_path, "r") as f:
            dual_recording_metadata = yaml.safe_load(f)
        return dual_recording_metadata["sync_ratio"]

    def _match_muscle_frames_to_behavior(
        self,
        muscle_path_by_muscle_frameid: dict[int, Path],
        behavior_frame_ids: list,
    ) -> tuple[dict[int, int], dict[int, Path]]:
        """Match muscle frames to behavior frames using sync ratio.

        Args:
            muscle_path_by_muscle_frameid: Dict mapping muscle frame IDs to paths.
            behavior_frame_ids: List of available behavior frame IDs.

        Returns:
            Tuple of (segmap_stack_indices_by_muscle_frameid, metadata_paths_by_muscle_frameid).
        """
        segmap_stack_idx_by_muscle_frameid = {}
        metadata_path_by_muscle_frameid = {}
        behavior_to_muscle_ratio = self._get_behavior_to_muscle_ratio()

        for muscle_frame_id in muscle_path_by_muscle_frameid.keys():
            behavior_frame_id = (muscle_frame_id + 1) * behavior_to_muscle_ratio
            if behavior_frame_id not in behavior_frame_ids:
                continue
            segmap_idx_in_stack = behavior_frame_ids.index(behavior_frame_id)
            segmap_stack_idx_by_muscle_frameid[muscle_frame_id] = segmap_idx_in_stack
            input_transform_metadata_path = (
                self.cropped_behavior_image_dir.parent.parent
                / "all"
                / f"frame_{behavior_frame_id:09d}.metadata.json"
            )
            metadata_path_by_muscle_frameid[muscle_frame_id] = (
                input_transform_metadata_path
            )

        n_matched_muscle_frames = len(segmap_stack_idx_by_muscle_frameid)
        n_total_muscle_frames = len(muscle_path_by_muscle_frameid)
        n_not_matched = n_total_muscle_frames - n_matched_muscle_frames
        logging.info(
            f"Found {n_matched_muscle_frames} muscle frames with corresponding "
            f"bodyseg masks. Skipped {n_not_matched} muscle frames "
            f"without matching segmaps (and therefore behavior frames)."
        )

        return segmap_stack_idx_by_muscle_frameid, metadata_path_by_muscle_frameid

    def _prepare_initial_mapping_inputs(
        self,
        segmap_by_muscle_frameid: dict[int, np.ndarray],
        muscle_path_by_muscle_frameid: dict[int, Path],
        metadata_path_by_muscle_frameid: dict[int, Path],
        padding: int,
        seg_labels: list[str],
    ) -> list[dict]:
        """Prepare input arguments for parallel processing of muscle frames.

        Returns:
            List of keyword arguments for each frame to be processed.
        """
        segs_to_visualize_indices = None
        plots_subdir = None
        if self.debug_plots_dir is not None:
            plots_subdir = self.debug_plots_dir / "initial_mapped_segmaps"
            plots_subdir.mkdir(parents=True, exist_ok=True)
            segs_to_visualize_names = [
                x for x in seg_labels if x[:2] in legs or x.endswith("Antenna")
            ]
            segs_to_visualize_indices = [
                i for i, x in enumerate(seg_labels) if x in segs_to_visualize_names
            ]
        self.segs_to_visualize_indices = segs_to_visualize_indices

        muscle_frame_ids = list(segmap_by_muscle_frameid.keys())
        input_kwargs_all_frames = []
        for muscle_frame_id in muscle_frame_ids:
            segmap = segmap_by_muscle_frameid[muscle_frame_id]
            muscle_image_path = muscle_path_by_muscle_frameid[muscle_frame_id]
            input_transform_metadata_path = metadata_path_by_muscle_frameid[
                muscle_frame_id
            ]
            is_first_frame = muscle_frame_id == muscle_frame_ids[0]
            viz_output_path = None
            if plots_subdir is not None:
                viz_output_path = plots_subdir / f"frame_{muscle_frame_id:06d}.jpg"

            input_kwargs_all_frames.append(
                {
                    "segmap": segmap,
                    "muscle_image_path": muscle_image_path,
                    "input_transform_metadata_path": input_transform_metadata_path,
                    "padding": padding,
                    "suppress_size_mismatch_warning": not is_first_frame,
                    "viz_output_path": viz_output_path,
                    "segs_to_visualize_indices": segs_to_visualize_indices,
                    "muscle_vrange": self.muscle_vrange,
                }
            )

        return input_kwargs_all_frames

    def _save_initial_mapping_results(
        self,
        f_out: h5py.File,
        muscle_frame_ids: list[int],
        input_kwargs_all_frames: list[dict],
        results_all_frames: list[dict],
        seg_labels: list[str],
    ) -> None:
        """Save initial mapping results to HDF5 file.

        Args:
            f_out: Output HDF5 file handle.
            muscle_frame_ids: List of muscle frame IDs.
            input_kwargs_all_frames: Input parameters for each frame.
            results_all_frames: Processing results for each frame.
            seg_labels: List of segmentation class labels.
        """
        for i, muscle_frame_id in enumerate(muscle_frame_ids):
            input_kwargs = input_kwargs_all_frames[i]
            results = results_all_frames[i]
            grp = f_out.create_group(f"muscle_frame_{muscle_frame_id:06d}")
            grp.create_dataset(
                "muscle_image_cropped",
                data=results["muscle_image_cropped"],
                compression="gzip",
                shuffle=True,
            )
            grp.create_dataset(
                "segmap_cropped",
                data=results["segmap_cropped"],
                compression="gzip",
                shuffle=True,
            )
            grp.attrs["muscle_image_path"] = str(
                input_kwargs["muscle_image_path"].absolute()
            )
            grp.attrs["input_transform_metadata_path"] = str(
                input_kwargs["input_transform_metadata_path"].absolute()
            )
            grp.attrs["has_non_background_feature"] = results[
                "has_non_background_feature"
            ]
            grp.attrs.update(results["cropping_metadata"])

        f_out.attrs["class_labels"] = seg_labels

    def _prepare_fine_alignment_inputs(
        self,
        search_limit: int,
        foreground_class_indices: list[int],
        muscle_vrange: tuple[int, int],
        viz_output_dir: Path | None = None,
    ) -> list[dict]:
        """Prepare input arguments for fine alignment processing.

        Args:
            search_limit: Maximum search offset for template matching.
            foreground_class_indices: Indices of classes to use as foreground.
            muscle_vrange: Min/max values for muscle image normalization.
            viz_output_dir: Optional directory for visualization output.

        Returns:
            List of keyword arguments for each frame to be processed.
        """
        input_kwargs_all_frames = []

        with h5py.File(self.output_path, "a") as f:
            for frame_key in f.keys():
                grp = f[frame_key]
                muscle_image = grp["muscle_image_cropped"][:]
                segmap = grp["segmap_cropped"][:]

                if viz_output_dir:
                    frame_id = int(frame_key.split("_")[-1])
                    viz_output_path = viz_output_dir / f"frame_{frame_id:06d}.jpg"
                else:
                    viz_output_path = None

                kwargs = {
                    "muscle_image": muscle_image,
                    "segmap": segmap,
                    "search_limit": search_limit,
                    "foreground_class_indices": foreground_class_indices,
                    "muscle_vrange": muscle_vrange,
                    "viz_output_path": viz_output_path,
                }
                input_kwargs_all_frames.append(kwargs)

        return input_kwargs_all_frames

    def _prepare_morph_denoising_inputs(
        self,
        morph_kernel: np.ndarray,
        n_iterations: int,
        viz_output_dir: Path | None = None,
    ) -> list[dict]:
        input_kwargs_all_frames = []
        with h5py.File(self.output_path, "a") as f:
            class_labels = list(f.attrs["class_labels"])
            for frame_key in f.keys():
                grp = f[frame_key]
                segmap = grp["segmap_fine_aligned"][:]
                muscle_image = grp["muscle_image_cropped"][:]

                if viz_output_dir:
                    frame_id = int(frame_key.split("_")[-1])
                    viz_output_path = viz_output_dir / f"frame_{frame_id:06d}.jpg"
                else:
                    viz_output_path = None

                kwargs = {
                    "segmap": segmap,
                    "class_labels": class_labels,
                    "morph_kernel": morph_kernel,
                    "n_iterations": n_iterations,
                    "muscle_vrange": self.muscle_vrange,
                    "viz_output_path": viz_output_path,
                    "muscle_image": muscle_image,
                    "segs_to_visualize_indices": self.segs_to_visualize_indices,
                }
                input_kwargs_all_frames.append(kwargs)

        return input_kwargs_all_frames


def _load_bodyseg_predictions(bodyseg_path: Path) -> tuple[list, np.ndarray, list]:
    """Load bodyseg predictions from HDF5 file.

    Args:
        bodyseg_path: Path to bodyseg output file.

    Returns:
        Tuple of (behavior_frame_ids, pred_segmaps, seg_labels).
    """
    logging.info(f"Loading bodyseg output from: {bodyseg_path}")
    with h5py.File(bodyseg_path, "r") as f_bodyseg:
        behavior_frame_ids = list(f_bodyseg["frame_ids"][:])
        pred_segmaps = f_bodyseg["pred_segmap"][:]
        seg_labels = list(f_bodyseg["pred_segmap"].attrs["class_labels"])
    logging.info(f"Bodyseg output contains {len(behavior_frame_ids)} frames")
    return behavior_frame_ids, pred_segmaps, seg_labels


def _get_foreground_class_ids(
    foreground_classes: list[str] | str, class_labels: list[str]
) -> list[int]:
    """Get class IDs corresponding to specified foreground classes.

    Args:
        foreground_classes: List of class names or "legs" for all leg segments.
        class_labels: List of all available class labels.

    Returns:
        List of class indices matching the foreground criteria.
    """
    foreground_class_ids = []
    for i, label in enumerate(class_labels):
        if foreground_classes == "legs" and label[:2] in legs:
            foreground_class_ids.append(i)
        elif isinstance(foreground_classes, list) and label in foreground_classes:
            foreground_class_ids.append(i)
    return foreground_class_ids


def _map_segmap_to_muscle_single_frame(
    segmap: np.ndarray,
    muscle_image_path: Path,
    input_transform_metadata_path: Path,
    padding: int,
    suppress_size_mismatch_warning: bool,
    viz_output_path: Path | None,
    segs_to_visualize_indices: list[int],
    muscle_vrange: tuple[int, int],
):
    """Process a single muscle frame for initial mapping.

    Args:
        segmap: Segmentation map from bodyseg model.
        muscle_image_path: Path to corresponding muscle image.
        input_transform_metadata_path: Path to alignment metadata.
        padding: Padding around bounding box for cropping.
        suppress_size_mismatch_warning: Whether to suppress size warnings.
        viz_output_path: Optional path to save visualization.
        segs_to_visualize_indices: Indices of segments to visualize.
        muscle_vrange: Min and max values for muscle visualization.

    Returns:
        Dictionary containing processed muscle image, segmentation map,
        and cropping metadata.
    """
    # Load muscle image
    # The postprocessing step in spotlight-tools has already warped muscle images to
    # the corresponding behavior images. So just load them here.
    muscle_image = imageio.imread(muscle_image_path)

    # Load metadata for this frame
    with open(input_transform_metadata_path, "r") as f:
        input_transform_metadata = json.load(f)

    # Check if the size of the muscle image matches the size of the "original" behavior
    # image that we are trying to transform the segmap back to
    size_before_transform = tuple(input_transform_metadata["rotation"]["input_size"])
    if size_before_transform != muscle_image.shape:
        if not suppress_size_mismatch_warning:
            logging.warning(
                f"Muscle image shape {muscle_image.shape} does not match target "
                f"(behavior) shape {size_before_transform}. This is likely because the "
                f"behavior camera cannot record at any arbitrary frame size, and it is "
                f"rounded to the nearest allowable size (multiples of 64). "
                f"Cropping the muscle image to the target size."
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
    size_after_crop = input_transform_metadata["crop"]["output_size"]
    zoom_factor = (
        size_after_crop[0] / segmap.shape[0],
        size_after_crop[1] / segmap.shape[1],
    )
    segmap_before_resize = ndimage.zoom(segmap, zoom_factor, order=0)

    # Undo alignment transformations (rotation + crop) to map segmap back to the
    # pixel space of raw Spotlight behavior images
    segmap_before_alignment = reverse_rotation_and_crop(
        segmap_before_resize,
        rotation_params=input_transform_metadata["rotation"],
        crop_params=input_transform_metadata["crop"],
        fill_value=0,
    )

    # Crop muscle image and segmap to a rough bounding box
    # There's no need storing the full image since the fly only occupies a small
    # portion. Crop the image based on the bounding box of non-background features,
    # but add some padding in case we want to slightly adjust the segmap in later
    # postprocessing steps.
    has_feature, muscle_image_cropped, segmap_cropped, cropping_metadata = (
        _crop_mapping_output_by_segmap_feature(
            muscle_image, segmap_before_alignment, padding
        )
    )

    mapping_results = {
        "has_non_background_feature": has_feature,
        "muscle_image_cropped": muscle_image_cropped,
        "segmap_cropped": segmap_cropped,
        "cropping_metadata": cropping_metadata,
    }

    # Visualize mapping results by drawing mask contours over muscle images (optional)
    if viz_output_path is not None:
        masks = []
        for seg_value in segs_to_visualize_indices:
            masks.append(segmap_cropped == seg_value)
        draw_mask_contours(
            image=muscle_image_cropped,
            masks=np.stack(masks, axis=0),
            muscle_vrange=muscle_vrange,
            output_path=viz_output_path,
        )

    return mapping_results


def _crop_mapping_output_by_segmap_feature(
    muscle_image, segmap_before_alignment, padding
) -> tuple[bool, np.ndarray, np.ndarray, dict[str, list[int]]]:
    """Crop muscle image and segmap to a rough bounding box around
    non-background features with some padding. The extent of the
    non-background features is returned in the metadata as
    `segmap_feature_bbox_*` and the actual crop box used is returned
    as `crop_box_*`.

    Args:
        muscle_image: np.ndarray, shape (H, W)
        segmap_before_alignment: np.ndarray, shape (H, W)
        padding: int, number of pixels to pad around the bounding box

    Returns:
        has_feature: bool, whether any non-background feature is found
        muscle_image_cropped: np.ndarray, shape (h, w)
        segmap_cropped: np.ndarray, shape (h, w)
        cropping_metadata: dict, contains the following keys:
            - "segmap_feature_bbox_row_range": [min_row, max_row]
            - "segmap_feature_bbox_col_range": [min_col, max_col]
            - "crop_box_row_range": [crop_row_min, crop_row_max]
            - "crop_box_col_range": [crop_col_min, crop_col_max]
    """
    non_background_mask = segmap_before_alignment > 0  # background label is 0

    if non_background_mask.sum() == 0:
        # If segmap contains only background, return the original images without
        # cropping. Use a flag to indicate no features are found.
        logging.warning(
            "Segmap contains only background. Returning original images without "
            "cropping. This should not happen - check upstream processing."
        )
        muscle_image_cropped = muscle_image
        segmap_cropped = segmap_before_alignment
        cropping_metadata = {
            "segmap_feature_bbox_row_range": [0, muscle_image.shape[0]],
            "segmap_feature_bbox_col_range": [0, muscle_image.shape[1]],
            "crop_box_row_range": [0, muscle_image.shape[0]],
            "crop_box_col_range": [0, muscle_image.shape[1]],
        }
        has_feature = False
        return has_feature, muscle_image_cropped, segmap_cropped, cropping_metadata

    ys, xs = np.where(non_background_mask)
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
    has_feature = True
    return has_feature, muscle_image_cropped, segmap_cropped, cropping_metadata


def _template_match_mask_to_muscle_single_frame(
    muscle_image: np.ndarray,
    segmap: np.ndarray,
    search_limit: int,
    foreground_class_indices: list[int],
    muscle_vrange: tuple[int, int],
    viz_output_path: Path | None,
):
    """
    Align a single binary mask to a muscle image using template matching.
    This function assumes that the only misalignment is a small translation
    (NOT affine).

    Args:
        muscle_image: Muscle image array.
        segmap: Segmentation map with class labels.
        search_limit: Maximum offset to search in any direction.
        foreground_class_indices: List of class indices to use as foreground.
        muscle_vrange: Min and max values for muscle image normalization.
        viz_output_path: Optional path to save visualization.

    Returns:
        Dictionary containing:
            - segmap_fine_aligned: Shifted segmentation map
            - x_shift: Optimal column-wise shift applied
            - y_shift: Optimal row-wise shift applied
            - corr_matrix: Correlation matrix from template matching
    """
    # Normalize muscle image
    muscle_image = muscle_image.astype(np.float32)
    vmin, vmax = muscle_vrange
    muscle_image_norm = np.clip((muscle_image - vmin) / (vmax - vmin), 0, 1)

    # Get foreground mask
    foreground_mask = np.isin(segmap, foreground_class_indices)

    # Find optimal shifts using template matching
    if muscle_image.shape != foreground_mask.shape:
        raise ValueError(
            f"Muscle image shape {muscle_image.shape} does not match "
            f"foreground mask shape {foreground_mask.shape}"
        )
    x_shift, y_shift, corr_matrix = _find_optimal_translation_by_template_matching(
        muscle_image_norm, foreground_mask, search_limit
    )
    segmap_fine_aligned = ndimage.shift(
        segmap,
        shift=(y_shift, x_shift),
        order=0,
        mode="nearest",
        cval=np.nan,
    )

    # Visualize alignment
    if viz_output_path is not None:
        draw_template_matching_viz(
            muscle_image=muscle_image,
            foreground_mask=foreground_mask,
            x_shift=x_shift,
            y_shift=y_shift,
            corr_matrix=corr_matrix,
            search_limit=search_limit,
            output_path=viz_output_path,
            muscle_vrange=muscle_vrange,
        )

    return {
        "segmap_fine_aligned": segmap_fine_aligned,
        "x_shift": x_shift,
        "y_shift": y_shift,
        "corr_matrix": corr_matrix,
    }


def _find_optimal_translation_by_template_matching(muscle_image, mask, search_limit=50):
    """
    Align a single binary mask to a muscle image using template matching.
    This function assumes that the only misalignment is a small translation
    (NOT affine).

    Args:
        muscle_image_norm (np.ndarray): Muscle image as normalized from
            vmin-vmax to 0-1.
        mask (np.ndarray): Boolean segmentation mask
        search_limit (int): Maximum offset to search in any direction

    Returns:
        x_shift: Optimal column-wise shifts to apply to the mask
        y_shift: Optimal row-wise shifts to apply to the mask
        correlation_matrix: Correlation matrix from template matching
    """
    # Convert to float32
    muscle_image = muscle_image.astype(np.float32)
    mask = mask.astype(np.float32)
    n_rows, n_cols = muscle_image.shape
    assert mask.shape == (n_rows, n_cols), "Mask and muscle image must have same shape"

    # Pad muscle image by search limit on all sides so that we can slide the mask on it
    padded_image = np.zeros(
        (n_rows + 2 * search_limit, n_cols + 2 * search_limit), dtype=np.float32
    )
    padded_image[
        search_limit : search_limit + n_rows, search_limit : search_limit + n_cols
    ] = muscle_image

    # Using mask as template, match it to padded muscle image using
    # normalized cross corrlation
    correlation_matrix = cv2.matchTemplate(padded_image, mask, cv2.TM_CCORR_NORMED)

    # Calculate shifts
    _, max_correlation, _, max_loc = cv2.minMaxLoc(correlation_matrix)
    x_shift = max_loc[0] - search_limit
    y_shift = max_loc[1] - search_limit

    return x_shift, y_shift, correlation_matrix


def _denoise_mask_for_each_class_single_frame(
    segmap: np.ndarray,
    class_labels: list[str],
    morph_kernel: int,
    n_iterations: int,
    muscle_vrange: tuple[int, int] | None = None,
    viz_output_path: Path | None = None,
    muscle_image: np.ndarray | None = None,
    segs_to_visualize_indices: list[int] | None = None,
):
    denoised_masks = []
    labels = []
    for i, class_label in enumerate(class_labels):
        mask = segmap == i
        if class_label.lower() == "background":
            denoised_mask = mask
        else:
            denoised_mask = _denoise_mask_by_morph(mask, morph_kernel, n_iterations)
        denoised_masks.append(denoised_mask)
        labels.append(class_label)
    denoised_masks = np.stack(denoised_masks, axis=0)

    if viz_output_path is not None:
        masks_for_viz = np.stack(
            [denoised_masks[i] for i in segs_to_visualize_indices], axis=0
        ).astype(np.bool_)
        draw_mask_contours(
            image=muscle_image,
            masks=masks_for_viz,
            muscle_vrange=muscle_vrange,
            output_path=viz_output_path,
        )

    return {"denoised_masks": denoised_masks, "labels": labels}


def _denoise_mask_by_morph(mask, morph_kernel, n_iterations):
    """
    Denoise a binary mask using morphological operations:
    1. Morphological opening to remove small objects
    2. Morphological closing to fill small holes
    3. Keep only the largest connected component

    Args:
        mask: np.ndarray, binary mask to denoise
        morph_kernel: Structuring element for morphological operations
        n_iterations: Number of iterations for opening and closing

    Returns:
        denoised_mask: np.ndarray, denoised binary mask
    """
    # Morphological opening
    opened_mask = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        morph_kernel,
        iterations=n_iterations,
    )

    # Morphological closing
    closed_mask = cv2.morphologyEx(
        opened_mask,
        cv2.MORPH_CLOSE,
        morph_kernel,
        iterations=n_iterations,
    )

    # Keep only the largest connected component
    num_labels, labels_im = cv2.connectedComponents(closed_mask.astype(np.uint8))
    if num_labels <= 1:
        # No foreground components found
        return np.zeros_like(mask, dtype=bool)

    largest_component = 1 + np.argmax(
        [np.sum(labels_im == label_id) for label_id in range(1, num_labels)]
    )
    denoised_mask = labels_im == largest_component

    return denoised_mask
