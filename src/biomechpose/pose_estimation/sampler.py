import torch
import numpy as np
import h5py
import cv2
import imageio.v2 as imageio
import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from typing import Callable

from biomechpose.util import (
    check_num_frames,
    read_frames_from_video,
    round_up_to_multiple,
    default_video_writing_ffmpeg_params,
)


class SimulatedDataSequence:
    def __init__(
        self,
        synthetic_video_paths: list[Path],
        simulated_labels_path: Path | None = None,
        sim_name: str = "",
        cache_metadata: bool = True,
        use_cached_metadata: bool = True,
    ):
        self.synthetic_video_paths = synthetic_video_paths
        self.simulated_labels_path = simulated_labels_path
        self.sim_name = sim_name

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


class SyntheticFramesSampler:
    """Sampler for contrastive pretraining using an InfoNCE/SimCLR-like
    loss. Each sample is a *batch* of `batch_size` frames drawn from
    `n_variants` videos (i.e. domain-translated versions of the same
    simulation using different style transfer models).

    Using one of the `n_variants` videos as the "anchor", the batch should
    contain (`n_variants` - 1) positive pairs (i.e. same frame translated
    into experimental domain using different style transfer models), and
    (`batch_size` * `n_variants` - `n_variants`) negative pairs. For the
    negative pairs, we enforce that all frames are at least
    `sampling_stride` frames apart in time.

    In theory, the number of samples that can be drawn from this dataset is
    combinatorial (Comb(`nframes`, `batch_size`) conditioned on all pairs
    of sampled values being at least `sampling_stride` frames apart). This
    results in an astronomically large number. Instead, we add an extra
    constraint that no frame can be sampled twice in each epoch, regardless
    of combination with other frames in the batch. For example, if
    `batch_size` is 4 `sampling_stride` is 3, we will reshape all possible
    global frame indices as follows:

                       |   offset_0 |   offset_1 |   offset_2
        ---------------+------------+------------+------------
         super_batch_0 |          0 |          1 |          2
                       |          3 |          4 |          5
                       |          6 |          7 |          8
                       |          9 |         10 |         11
        ---------------+------------+------------+------------
         super_batch_1 |         12 |         13 |         14
                       |         15 |         16 |         17
                       |         18 |         19 |         20
                       |         21 |         22 |         23

    Here, frames are divided into "super-batches" of size
    (`batch_size` * `sampling_stride`). Each super-batch is further divided
    into `sampling_stride` "offsets," each of size `batch_size`. Therefore,
    this partitions the whole dataset into
    floor(`n_frames_total` / (`batch_size` * `sampling_stride`)) * `sampling_stride`
    non-overlapping samples, each containing `batch_size` frames such that
    every two frames are at least `sampling_stride` apart in time. The
    remainder of the frames that do not fit into a full super-batch are
    discarded. (Note: This actually results in a smaller number of possible
    samples because the minimum time difference does not apply when the two
    frames are drawn from different simulations.)

    When `__getitem__` is called, this dataset returns a Torch tensor of
    shape (n_variants, batch_size, n_channels, img_height, img_width).
    """

    def __init__(
        self,
        simulated_data_sequences: list[SimulatedDataSequence],
        batch_size: int,
        sampling_stride: int = 10,
        transform: Callable | None = None,
        load_labels: bool = True,
    ):
        """Initializes ContrastivePretrainingDataset.

        Args:
            simulated_data_sequences (list[SimulatedDataSequence]): List of
                SimulatedDataSequence objects containing the synthetic videos
                and simulation data to sample from.
            batch_size (int): Number of frame samples per batch.
            sampling_stride (int, optional): Minimum time difference (in
                frames) between two samples drawn from the same video
                for them to be considered a negative pair. Defaults to 10.
            transform (Callable, optional): Transform to be applied on a
                sample before returning it. If a transform is specified and
                load_labels if False, this callable should map batch of
                images of shape (N, C, H, W) to a batch of images of shape
                (N, C, H, W) - though N, C, H, W may have different values.
                If a transform is specified and load_labels if True, this
                callable should map a tuple of (images, labels) to a tuple
                of (images, labels), where both the images and the labels
                are transformed appropriately. Before the transform, the
                SimulatedDataSequence class will have already made the
                frames monochrome, and this sampler will already have
                converted uint8 numpy arrays of shape (H, W, C) to float32
                torch tensors of shape (C, H, W). The transform should
                expect these converted types and shapes. Defaults to None.
            load_labels (bool, optional): Whether to load labels from
                simulation data files. If True, __getitem__ will return a
                (images, labels) tuple instead of just images. Practically,
                use load_labels=False for unsupervised (pre)training and
                use load_labels=True for supervised pose estimation.
                Defaults to True.
        """
        self.simulated_data_sequences = simulated_data_sequences
        self.batch_size = batch_size
        self.sampling_stride = sampling_stride
        self.transform = transform
        self.load_labels = load_labels

        assert self.batch_size > 1, "`batch_size` must be > 1"
        assert self.sampling_stride >= 1, "`sampling_stride` must be >= 1"
        assert len(simulated_data_sequences) > 0, "At least 1 simulation required"

        # Check number of variants per frame
        assert (
            len(set([seq.n_variants for seq in simulated_data_sequences])) == 1
        ), "All simulations must have the same number of variants"
        self.n_variants = simulated_data_sequences[0].n_variants

        # Check image size and FPS
        _image_sizes = set()
        _fpss = set()
        for seq in simulated_data_sequences:
            _image_sizes.add(seq.frame_size)
            _fpss.add(seq.fps)
        assert len(_image_sizes) == 1, "All simulations must have the same image size"
        assert len(_fpss) == 1, "All simulations must have the same FPS"
        self.frame_size = _image_sizes.pop()
        self.fps = _fpss.pop()

        # Check number of frames per simulation and in total
        self.n_frames_per_sim = np.array(
            [seq.n_frames for seq in simulated_data_sequences]
        )
        self._start_global_ids_per_sim = np.zeros(
            len(simulated_data_sequences), dtype=np.int32
        )
        self._start_global_ids_per_sim[1:] = np.cumsum(self.n_frames_per_sim)[:-1]

        # Lookup table mapping from global frame index to simulation ID
        n_frames_total = np.sum(self.n_frames_per_sim)  # ~4MB of RAM for 1h of data
        self._sim_id_by_global_idx = np.zeros(n_frames_total, dtype=np.int32)
        start_idx = 0
        for sim_id, num_frames in enumerate(self.n_frames_per_sim):
            self._sim_id_by_global_idx[start_idx : start_idx + num_frames] = sim_id
            start_idx += num_frames
        assert start_idx == n_frames_total, "Error in n_frames counting"

        # Calculate length of dataset
        n_super_batches = n_frames_total // (self.batch_size * self.sampling_stride)
        if n_super_batches == 0:
            raise ValueError(
                "Not enough frames to construct one batch. Try including more "
                "simulation data, reducing `batch_size`, or reducing `sampling_stride`."
            )
        self.dataset_length = int(n_super_batches) * self.sampling_stride
        n_frames_total_trunc = n_super_batches * self.batch_size * self.sampling_stride
        assert n_frames_total_trunc <= n_frames_total, "Error in n_frames counting"
        self.total_n_frames = n_frames_total_trunc

        # Pre-compute frame indices for each sample in the dataset
        self._frame_indices = np.arange(
            0, n_frames_total_trunc, dtype=np.int32
        ).reshape(n_super_batches, self.batch_size, self.sampling_stride)

    def __len__(self) -> int:
        return self.dataset_length

    def determine_batch_frame_ids(
        self, idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the frame indices that should be sampled to construct
        the batch at index `idx`.

        Returns:
            - global_frame_ids: (batch_size,) np.ndarray of int
                Global frame indices for each sample in the batch.
            - sim_ids: (batch_size,) np.ndarray of int
                Indices indicating which simulation each frame should be
                drawn from.
            - local_frame_ids: (batch_size,) np.ndarray of int
                indicates which the index of the frame within the
                respective simulation.
        """
        # Get global indices of sampled frames
        super_batch_idx = idx // self.sampling_stride
        offset = idx % self.sampling_stride
        # global_frame_ids: (batch_size,)
        global_frame_ids = self._frame_indices[super_batch_idx, :, offset]
        # sampled_sim_ids: (batch_size,), dtype=int
        sim_ids = self._sim_id_by_global_idx[global_frame_ids]
        # local_frame_ids: (batch_size,), dtype=int
        local_frame_ids = np.zeros_like(sim_ids)

        # Iterate over simulations involved and compute local frame IDs
        for this_sim_id in np.unique(sim_ids):
            # mask: (batch_size,), dtype=bool
            mask = sim_ids == this_sim_id
            # global_frame_ids_this_sim: (n_frames_this_sim,), dtype=int
            global_frame_ids_this_sim = global_frame_ids[mask]
            # start_idx_this_sim: scalar, dtype=int
            start_idx_this_sim = self._start_global_ids_per_sim[this_sim_id]
            # local_frame_ids[mask]: (n_frames_this_sim,), dtype=int
            local_frame_ids[mask] = global_frame_ids_this_sim - start_idx_this_sim

        return global_frame_ids, sim_ids, local_frame_ids

    def __getitem__(self, idx: int) -> torch.Tensor:
        """If self.load_labels is False, returns:
            - frames: (n_variants, batch_size, n_channels, img_height, img_width)
                torch tensor of float32 in [0, 1]

        If self.load_labels is True, returns a tuple of:
            - frames: (n_variants, batch_size, n_channels, img_height, img_width)
                torch tensor of float32 in [0, 1]
            - labels: dict of torch tensors, each of shape (batch_size, ...)
        """
        # Get frame IDs to be sampled
        global_frame_ids, sim_ids, local_frame_ids = self.determine_batch_frame_ids(idx)
        # sim_ids should be monotonically increasing
        assert np.diff(sim_ids).min() >= 0, "Error in sim_ids computation"

        # Load frames and labels
        # frames_by_sim: list of (n_variants, n_frames_this_sim, H, W) arrays
        frames_by_sim = []
        # labels_by_sim: each dict value is a list of (n_frames_this_sim, ...) arrays
        labels_by_sim = defaultdict(list)

        # Read the appropriate subset of data from each simulation involved
        for sim_id in np.unique(sim_ids):
            mask = sim_ids == sim_id
            local_frame_ids_this_sim = local_frame_ids[mask].tolist()
            sim = self.simulated_data_sequences[sim_id]
            frames = sim.read_synthetic_frames(local_frame_ids_this_sim)
            frames_by_sim.append(frames)
            if self.load_labels:
                labels = sim.read_simulated_labels(local_frame_ids_this_sim)
                for k, v in labels.items():
                    labels_by_sim[k].append(v)

        # Combine data from all simulations into single arrays and convert to tensor
        frames = np.concatenate(frames_by_sim, axis=1)
        frames = torch.from_numpy(frames / 255.0).to(torch.float32)
        frames = frames[:, :, None, :, :]  # add channel dim
        if self.load_labels:
            labels = {
                k: torch.from_numpy(np.concatenate(v, axis=0))
                for k, v in labels_by_sim.items()
            }

        # Apply transform if specified
        if self.transform:
            if self.load_labels:
                frames, labels = self.transform(frames, labels)
            else:
                frames = self.transform(frames)

        if self.load_labels:
            return frames, labels
        else:
            return frames


def save_atomic_batch_frames(
    atomic_batch: torch.Tensor, output_path: Path, fps: int, spacing: int = 10
):
    """Write an atomic batch of frames to a video file.

    Args:
        atomic_batch (torch.Tensor): Tensor of shape (n_variants, n_frames,
            n_channels, height, width)
        output_path (Path): Path to save the video file.
        fps (int): Frames per second for the output video.
        spacing (int, optional): Number of pixels to insert between
            variants. Defaults to 10.
    """
    atomic_batch = atomic_batch.numpy()
    n_variants, n_frames, n_channels, n_rows, n_cols = atomic_batch.shape

    # Check n_channels and expand if needed
    if n_channels == 1:
        atomic_batch = np.repeat(atomic_batch, 3, axis=2)
        n_channels = 3
    elif n_channels == 3:
        pass  # OK
    else:
        raise ValueError(
            "atomic_batch must have shape "
            "(n_variants, n_frames, n_channels, height, width) and "
            "n_channels must be 1 or 3"
        )

    # Write to video
    n_cols_total = (n_cols * n_variants) + (n_variants - 1) * spacing
    n_cols_total = round_up_to_multiple(n_cols_total, 16)  # for video encoding
    n_rows_total = round_up_to_multiple(n_rows, 16)  # for video encoding
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=10,
        ffmpeg_params=default_video_writing_ffmpeg_params,
    ) as video_writer:
        for frame_idx in range(n_frames):
            image = np.zeros((n_rows_total, n_cols_total, n_channels), dtype=np.uint8)
            for variant_idx in range(atomic_batch.shape[0]):
                start_col = variant_idx * (n_cols + spacing)
                end_col = start_col + n_cols
                selection = atomic_batch[variant_idx, frame_idx, :, :, :]
                # Move channel to the last dimension
                selection = selection.transpose(1, 2, 0)
                selection = (selection * 255).astype(np.uint8)
                image[:n_rows, start_col:end_col, :] = selection
            video_writer.append_data(image.squeeze())


def load_atomic_batch_frames(
    video_path: Path,
    n_variants: int,
    image_size: tuple[int, int],
    n_channels: int = 1,
    spacing: int = 10,
) -> np.ndarray:
    """Read an atomic batch of frames from a video file.

    Args:
        video_path (Path): Path to the video file.
        n_variants (int): Number of variants in the atomic batch.
        image_size (tuple[int, int]): (height, width) of each variant.
        n_channels (int, optional): Number of image channels. Defaults to 1.
        spacing (int, optional): Number of pixels inserted between
            variants. Defaults to 10.

    Returns:
        atomic_batch (np.ndarray): Array of shape (n_variants, n_frames,
            n_channels, height, width)
    """
    frames, fps = read_frames_from_video(video_path)
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError(f"Error: no frames found in video {video_path}")
    n_rows_total, n_cols_total = frames[0].shape[:2]
    n_rows, n_cols = image_size
    assert (
        n_variants * n_cols + (n_variants - 1) * spacing <= n_cols_total
    ), "Error: width of concatenated video does not match expectation"

    atomic_batch = np.zeros(
        (n_variants, n_frames, n_channels, n_rows, n_cols), dtype=np.float32
    )
    for frame_idx, frame in enumerate(frames):
        for variant_idx in range(n_variants):
            start_col = variant_idx * (n_cols + spacing)
            end_col = start_col + n_cols
            selection = frame[:n_rows, start_col:end_col]
            if n_channels == 1:
                selection = selection[:, :, 0]  # only 1 channel (grayscale)
                selection = selection[..., None]  # add channel dim
            # Move the channel dimension to the front
            selection = selection.transpose(2, 0, 1)
            selection = np.ascontiguousarray(selection)
            # Convert to float32 and normalize to [0, 1]
            selection = selection / 255.0
            atomic_batch[variant_idx, frame_idx, :, :, :] = selection

    return torch.from_numpy(atomic_batch).to(torch.float32)


def save_atomic_batch_sim_data(
    sim_data: dict[str, np.ndarray],
    output_path: Path,
    metadata: dict | None = None,
):
    """Save simulation data for an atomic batch to an HDF5 file.

    Args:
        sim_data (dict[str, np.ndarray]): Dictionary of simulation data.
            Each value should be an array of shape (n_frames, ...).
        output_path (Path): Path to save the HDF5 file.
        metadata (dict, optional): Additional metadata to save as
            attributes in the HDF5 file. Defaults to None.
    """
    with h5py.File(output_path, "w") as f:
        for key, value in sim_data.items():
            compression = "lzf" if key == "body_seg_maps" else None
            f.create_dataset(key, data=value, compression=compression)
        f.attrs["n_frames"] = next(iter(sim_data.values())).shape[0]
        if metadata is not None:
            for key, value in metadata.items():
                f.attrs[key] = value


def load_atomic_batch_sim_data(input_path: Path, keys: list[str] | None) -> dict[str, np.ndarray]:
    """Load simulation data for an atomic batch from an HDF5 file.

    Args:
        input_path (Path): Path to the HDF5 file.
        keys (list[str] | None): List of keys to load. If None, all keys are loaded.

    Returns:
        dict[str, np.ndarray]: Dictionary of simulation data.
    """
    with h5py.File(input_path, "r") as f:
        if keys is None:
            keys = list(f.keys())
        else:
            for key in keys:
                if key not in f:
                    raise KeyError(f"Key {key} not found in file {input_path}")
            
        sim_data = {
            key: torch.from_numpy(f[key][:]).to(torch.float32) for key in keys
        }
        
        return sim_data
