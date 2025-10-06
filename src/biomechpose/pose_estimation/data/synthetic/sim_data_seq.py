import torch
import numpy as np
import h5py
import cv2
import json
from pathlib import Path
from typing import Iterator

from biomechpose.util import check_num_frames, read_frames_from_video


class SimulatedDataSequence:
    def __init__(
        self,
        synthetic_video_paths: list[Path],
        simulated_labels_path: Path | None = None,
        sim_name: str = "",
        cache_metadata: bool = True,
        use_cached_metadata: bool = True,
        original_image_size: tuple[int, int] | None = None,
    ):
        self.synthetic_video_paths = synthetic_video_paths
        self.simulated_labels_path = simulated_labels_path
        self.sim_name = sim_name
        self.original_image_size = original_image_size

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

        self.n_frames, self.frame_size, self.fps = self._get_metadata(
            synthetic_video_paths[0], cache_metadata, use_cached_metadata
        )

        # If image has been downsampled from the original, compute the zoom factor and
        # return keypoint positions converted to the scale of output images.
        if original_image_size is not None:
            zoom_factor = np.array(self.frame_size) / np.array(original_image_size)
        else:
            zoom_factor = np.array([1.0, 1.0])
        self.image_zoom_factor = zoom_factor

    def _get_metadata(self, sample_video_path, cache_metadata, use_cached_metadata):
        # Already checked that all videos are under the same directory, so just save the
        # cache there
        cache_path = sample_video_path.parent / "cached_sim_metadata.json"
        if use_cached_metadata and cache_path.is_file():
            try:
                with open(cache_path, "r") as f:
                    metadata = json.load(f)
                n_frames = metadata["n_frames"]
                frame_size = tuple(metadata["frame_size"])
                fps = metadata["fps"]
            except Exception as e:
                print(f"Corrupted metadata cache file {cache_path}")
                raise e
        else:
            n_frames = check_num_frames(sample_video_path)
            sample_frames, fps = read_frames_from_video(
                sample_video_path, frame_indices=[0]
            )
            frame_size = sample_frames[0].shape[:2]

            if cache_metadata:
                metadata = {
                    "n_frames": n_frames,
                    "frame_size": list(frame_size),
                    "fps": fps,
                }
                with open(cache_path, "w") as f:
                    json.dump(metadata, f, indent=2)
        return n_frames, frame_size, fps

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
        for i, variant_idx in enumerate(variant_indices):
            video_path = self.synthetic_video_paths[variant_idx]
            frames, fps = read_frames_from_video(video_path, frame_indices)
            for j, frame in enumerate(frames):
                data[i, j, :, :] = frame[:, :, 0]  # only 1 channel (grayscale)
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
                # Rescale to match output image size if original size is different
                if self.original_image_size is not None:
                    keypoint_pos[:, :, :2] *= self.image_zoom_factor[None, None, :]
                labels["keypoint_pos"] = keypoint_pos

            if load_mesh_states:
                raise NotImplementedError(
                    "Mesh states tracking (xyz + quat for 3D rotation) has not been "
                    "implemented yet"
                )

            if load_body_seg_maps:
                seg_labels_ds = ds["segmentation_labels"]
                # Resize to shape of synthetic frames via nearest neighbor resampling
                resized_body_seg_maps = np.empty(
                    (len(frame_indices), *self.frame_size), dtype=np.uint8
                )
                for i, frame_idx in enumerate(frame_indices):
                    input_map = seg_labels_ds[frame_idx, :, :]
                    resized_body_seg_maps[i, :, :] = cv2.resize(
                        input_map,
                        (self.frame_size[1], self.frame_size[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
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
                if "keypoint_pos" in labels:
                    labels["keypoint_pos"][:, :, :2] *= (256/464)  # TODO: remove this hack

            yield frames, labels
