import torch
import numpy as np
import h5py
import cv2
from pathlib import Path
from typing import Iterator
from pvio.io import get_video_metadata, read_frames_from_video


class SimulatedDataSequence:
    def __init__(
        self,
        synthetic_video_paths: list[Path],
        simulated_labels_path: Path | None = None,
        sim_name: str = "",
        cache_metadata: bool = True,
        use_cached_metadata: bool = True,
        original_image_size: tuple[int, int] | None = None,
        target_image_size: tuple[int, int] | None = None,
    ):
        """
        Args:
            ...
            original_image_size: (H, W) source content size for this simulation
                (i.e. the MuJoCo camera resolution). Used as the calibration of
                the keypoint and segmentation map coordinate space, and to strip
                any FFMPEG codec padding from the loaded video frames. Auto-
                derived from the segmentation labels shape if not provided.
            target_image_size: If set, frames, segmentation maps, and keypoint
                xy are returned at this (H, W). Frames are cropped to
                `original_image_size` (removing codec padding) then resized
                with INTER_AREA; segmentation maps are resized from
                `original_image_size` with INTER_NEAREST; keypoint xy are
                scaled by `target / original` (depth untouched). When set,
                ``self.frame_size`` reports the target size. When None,
                behavior is unchanged (frames returned at the raw video size).
        """
        self.synthetic_video_paths = synthetic_video_paths
        self.simulated_labels_path = simulated_labels_path
        self.sim_name = sim_name
        self.original_image_size = original_image_size
        self.target_image_size = target_image_size

        # Validate input paths
        for path in synthetic_video_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Video file {path} does not exist")
        if simulated_labels_path is not None and not simulated_labels_path.is_file():
            raise FileNotFoundError(
                f"Simulation data file {simulated_labels_path} does not exist"
            )
        if len(set([str(path.parent) for path in synthetic_video_paths])) != 1:
            raise ValueError("All synthetic videos must be in the same directory")

        # Check number of variants of synthetic videos
        self.n_variants = len(synthetic_video_paths)
        assert self.n_variants > 0, "At least one synthetic video required"

        metadata = get_video_metadata(
            synthetic_video_paths[0], cache_metadata, use_cached_metadata
        )
        self.n_frames = metadata["n_frames"]
        self._video_frame_size = metadata["frame_size"]  # raw video (may be codec-padded)
        self.fps = metadata["fps"]

        # If original_image_size was not provided manually, extract it from the segmentation labels
        if self.original_image_size is None and simulated_labels_path is not None:
            with h5py.File(simulated_labels_path, "r") as ds:
                seg_shape = ds["postprocessed"]["segmentation_labels"].shape
                self.original_image_size = (seg_shape[1], seg_shape[2])

        # If there's a mismatch between the original image size and the video frame size,
        # we assume it is due to FFMPEG padding the output video to a multiple of 16.
        # We will pad the segmentation maps to match, and leave keypoints unscaled.

        # When target_image_size is set, all returned frames / seg maps / xy live
        # in target-pixel space. We need original_image_size to do the rescaling.
        if self.target_image_size is not None:
            if self.original_image_size is None:
                raise ValueError(
                    "target_image_size was specified but original_image_size could not be "
                    "determined (no simulated_labels_path and no manual value). Pass "
                    "original_image_size explicitly."
                )
            self.frame_size = tuple(self.target_image_size)
        else:
            self.frame_size = self._video_frame_size

    def get_sim_data_metadata(self) -> dict:
        metadata = {}
        with h5py.File(self.simulated_labels_path, "r") as ds:
            postprocessed_ds = ds["postprocessed"]
            metadata["dof_angles"] = dict(postprocessed_ds["dof_angles"].attrs)
            metadata["keypoint_pos"] = dict(postprocessed_ds["keypoint_pos"].attrs)
            metadata["segmentation_labels"] = dict(
                postprocessed_ds["segmentation_labels"].attrs
            )
        return metadata

    def __len__(self) -> int:
        return self.n_frames

    def _check_frame_indices_validity(self, frame_indices: list[int]) -> bool:
        if len(frame_indices) != len(set(frame_indices)):
            raise ValueError("Requested frame indices must not contain duplicates")
        if min(frame_indices) < 0 or max(frame_indices) >= self.n_frames:
            raise ValueError(f"Requested frame indices out of range")
        return True

    def _check_variant_indices_validity(self, variant_indices: list[int]) -> bool:
        if len(variant_indices) != len(set(variant_indices)):
            raise ValueError("Requested variant indices must not contain duplicates")
        if min(variant_indices) < 0 or max(variant_indices) >= self.n_variants:
            raise ValueError(f"Requested variant indices out of range")
        return True

    def read_synthetic_frames(
        self, frame_indices: list[int], variant_indices: list[int] | None = None
    ) -> np.ndarray:
        """Reads specified frames from specified variants of the synthetic
        videos.

        Args:
            frame_indices (list[int]): List of frame indices to read.
            variant_indices (list[int], optional): List of variant indices
                to read. Each index should be in [0, n_variants - 1]. If
                None, read all variants. Defaults to None.

        Returns:
            np.ndarray: Array of dtype uint8 and shape
                (len(variant_indices), len(frame_indices), img_height,
                img_width) containing the requested variants of the
                requested frames.
        """
        if variant_indices is None:
            variant_indices = list(range(self.n_variants))
        self._check_frame_indices_validity(frame_indices)
        self._check_variant_indices_validity(variant_indices)

        data = np.empty(
            (len(variant_indices), len(frame_indices), *self.frame_size), dtype=np.uint8
        )
        do_resize = self.target_image_size is not None
        if do_resize:
            orig_h, orig_w = self.original_image_size
            target_h, target_w = self.target_image_size

        for i, variant_idx in enumerate(variant_indices):
            video_path = self.synthetic_video_paths[variant_idx]
            frames, fps = read_frames_from_video(video_path, frame_indices)
            for j, frame in enumerate(frames):
                gray = frame[:, :, 0]  # only 1 channel (grayscale)
                if do_resize:
                    # Crop to original_image_size (strip FFMPEG codec padding),
                    # then resize to target_image_size with INTER_AREA (good
                    # for downsampling).
                    gray = gray[:orig_h, :orig_w]
                    if (orig_h, orig_w) != (target_h, target_w):
                        gray = cv2.resize(
                            gray,
                            (target_w, target_h),  # cv2 takes (W, H)
                            interpolation=cv2.INTER_AREA,
                        )
                data[i, j, :, :] = gray
        return data

    def read_simulated_labels(
        self,
        frame_indices: list[int],
        *,
        load_dof_angles: bool = True,
        load_keypoint_pos: bool = True,
        load_mesh_states: bool = False,
        load_body_seg_maps: bool = True,
    ) -> dict[str, np.ndarray]:
        self._check_frame_indices_validity(frame_indices)

        labels = {}
        with h5py.File(self.simulated_labels_path, "r") as ds:
            ds = ds["postprocessed"]

            if load_dof_angles:
                labels["dof_angles"] = ds["dof_angles"][frame_indices, :]

            if load_keypoint_pos:
                keypoint_pos_ds = ds["keypoint_pos/camera_coords"]
                keypoint_pos = keypoint_pos_ds[frame_indices, :, :]
                assert (
                    len(keypoint_pos.shape) == 3 and keypoint_pos.shape[2] == 3
                ), f"Unexpected keypoint_pos shape: {keypoint_pos.shape}"
                # Scale xy when an extraction-time resize is in effect. Depth
                # (index 2) is in mm and must NOT be scaled.
                if self.target_image_size is not None:
                    orig_h, orig_w = self.original_image_size
                    target_h, target_w = self.target_image_size
                    scale_x = target_w / orig_w
                    scale_y = target_h / orig_h
                    keypoint_pos = keypoint_pos.astype(np.float32, copy=True)
                    keypoint_pos[..., 0] *= scale_x
                    keypoint_pos[..., 1] *= scale_y
                labels["keypoint_pos"] = keypoint_pos

            if load_mesh_states:
                raise NotImplementedError(
                    "Mesh states tracking (xyz + quat for 3D rotation) has not been "
                    "implemented yet"
                )

            if load_body_seg_maps:
                seg_labels_ds = ds["segmentation_labels"]

                target_frame_size = self.frame_size
                resized_body_seg_maps = np.empty(
                    (len(frame_indices), *target_frame_size), dtype=np.uint8
                )

                if self.target_image_size is not None:
                    # Resize from original_image_size to target_image_size using
                    # nearest-neighbor (seg maps are class indices).
                    out_h, out_w = self.target_image_size
                    orig_h, orig_w = self.original_image_size
                    same_size = (out_h, out_w) == (orig_h, orig_w)
                    for i, frame_idx in enumerate(frame_indices):
                        input_map = seg_labels_ds[frame_idx, :, :]
                        # Defensively crop in case the stored map is larger than
                        # the declared original_image_size.
                        input_map = input_map[:orig_h, :orig_w]
                        if same_size:
                            resized_body_seg_maps[i, :, :] = input_map
                        else:
                            resized_body_seg_maps[i, :, :] = cv2.resize(
                                input_map,
                                (out_w, out_h),  # cv2 takes (W, H)
                                interpolation=cv2.INTER_NEAREST,
                            )
                else:
                    # Original behavior: optionally pad seg maps when the video
                    # was FFMPEG-padded above original_image_size.
                    pad_bottom = 0
                    pad_right = 0
                    if self.original_image_size is not None:
                        pad_bottom = max(0, target_frame_size[0] - self.original_image_size[0])
                        pad_right = max(0, target_frame_size[1] - self.original_image_size[1])

                    for i, frame_idx in enumerate(frame_indices):
                        input_map = seg_labels_ds[frame_idx, :, :]
                        if pad_bottom > 0 or pad_right > 0:
                            resized_body_seg_maps[i, :, :] = cv2.copyMakeBorder(
                                input_map, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0
                            )
                        else:
                            resized_body_seg_maps[i, :, :] = input_map

                labels["body_seg_maps"] = resized_body_seg_maps

        return labels

    def generate_batches(self, batch_size: int) -> Iterator[torch.Tensor]:
        """Generator that yields batches of synthetic images as PyTorch tensors.

        Args:
            batch_size (int): Total batch size (will be divided by n_variants)

        Yields:
            tuple[torch.Tensor, dict[str, torch.Tensor] | None]: A tuple containing:
                - frames: Batch tensor of shape (batch_size, 3, height, width)
                  with values normalized to [0, 1]
                - labels: Dictionary of label tensors if available, otherwise None
        """
        n_samples_per_batch = batch_size // self.n_variants
        n_batches = (self.n_frames + n_samples_per_batch - 1) // n_samples_per_batch

        for i in range(n_batches):
            start_idx = i * n_samples_per_batch
            end_idx = min((i + 1) * n_samples_per_batch, self.n_frames)
            frame_ids = list(range(start_idx, end_idx))

            # Load frames
            frames: np.ndarray = self.read_synthetic_frames(frame_ids)  # not tensor!
            # Change to torch tensor convention:
            # 1. n_channels before H and W, a single collapsed batch dimension in front
            #    (n_variants * n_frames, n_channels=3, n_rows, n_cols)
            # 2. Convert uint8 numpy array to float32 torch tensor normalized to [0, 1]
            n_variants, n_frames, n_rows, n_cols = frames.shape
            frames = torch.from_numpy(frames)
            frames = frames.to(dtype=torch.float32) / 255.0
            frames = frames.view(n_variants * n_frames, 1, n_rows, n_cols)
            frames = frames.repeat(1, 3, 1, 1)

            # Load labels if required
            if self.simulated_labels_path is None:
                labels = None
            else:
                labels = self.read_simulated_labels(frame_ids)
                labels = {key: torch.from_numpy(value) for key, value in labels.items()}

            yield frames, labels
