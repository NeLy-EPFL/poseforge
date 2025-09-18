import torch
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from tqdm import tqdm

from biomechpose.util import (
    check_num_frames,
    read_frames_from_video,
    default_video_writing_ffmpeg_params,
)


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
        video_paths_by_sim_names: dict[str, list[Path]],
        batch_size: int,
        sampling_stride: int = 10,
        transform=None,
        n_channels: int = 3,
        cache_sim_length: bool = True,
        use_cached_sim_length: bool = True,
    ):
        """Initializes ContrastivePretrainingDataset.

        Args:
            video_paths_by_sim_names (dict[str, list[Path]]): Mapping from
                each simulation ID to a list of video file paths. Each
                video is a *variant* of the domain-translated simulation
                rendering (i.e. each video is the output of one of the
                ensemble of style transfer models). All simulations must
                have the same number of output variants.
            batch_size (int): Number of frame samples per batch.
            sampling_stride (int, optional): Minimum time difference (in
                frames) between two samples drawn from the same video
                for them to be considered a negative pair. Defaults to 10.
            transform (callable, optional): Transform to be applied on a
                sample. This dataset will convert uint8 numpy arrays of
                shape (H, W, C) to float32 torch tensors of shape (C, H, W)
                already---these do not need to be included in `transform`.
            n_channels (int, optional): Number of image channels. If
                `transform` includes a reduction transformation (e.g.
                grayscale conversion, or using only the 0th channel), this
                should be set to 1. If RGB, this should be set to 3.
        """
        self.video_paths_by_sim_names = video_paths_by_sim_names
        self.batch_size = batch_size
        self.sampling_stride = sampling_stride
        self.transform = transform
        self.n_channels = n_channels
        self.all_sim_names = list(video_paths_by_sim_names.keys())
        self.n_variants = len(video_paths_by_sim_names[self.all_sim_names[0]])

        # Check image size
        sample_video_path = video_paths_by_sim_names[self.all_sim_names[0]][0]
        sample_frames, self.fps = read_frames_from_video(
            sample_video_path, frame_indices=[0]
        )
        sample_frame = sample_frames[0]
        self.frame_size = sample_frame.shape[:2]  # (H, W)

        # Check number of frames per simulation
        self.n_frames_per_sim = self._get_sim_lengths(
            video_paths_by_sim_names, cache_sim_length, use_cached_sim_length
        )
        self._start_global_ids_per_sim = np.zeros(
            len(self.all_sim_names), dtype=np.int32
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
                "Error: Not enough frames to construct one batch. Try including more "
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

    @staticmethod
    def _get_sim_lengths(
        video_paths_by_sim_names: dict[str, list[Path]],
        cache_sim_length: bool,
        use_cached_sim_length: bool,
    ) -> np.ndarray:
        """Get number of frames in each simulation. This process involves
        reading the metadata of one video per simulation and can take some
        time, so implement a caching mechanism (the length of the
        simulation in numbers of frames is saved under a `n_frames.txt`
        file)."""
        n_frames_per_sim = np.zeros(len(video_paths_by_sim_names), dtype=np.int32)
        for sim_idx, paths in tqdm(
            enumerate(video_paths_by_sim_names.values()),
            disable=None,
            total=len(video_paths_by_sim_names),
        ):
            if use_cached_sim_length:
                cache_path = paths[0].parent / "n_frames.txt"
                if cache_path.is_file():
                    with open(cache_path, "r") as f:
                        try:
                            n_frames_per_sim[sim_idx] = int(f.read().strip())
                            continue
                        except ValueError:
                            # Cache file corrupted - proceed linearly to the next case
                            pass

            # If reached here, either not using cached length or cache file unavailable
            sim_length = check_num_frames(paths[0])
            n_frames_per_sim[sim_idx] = sim_length
            if cache_sim_length:
                cache_path = paths[0].parent / "n_frames.txt"
                with open(cache_path, "w") as f:
                    f.write(str(sim_length))
        return n_frames_per_sim

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
        """Returns a *batch* of samples with shape
        (n_variants, batch_size, n_channels, img_height, img_width).
        """
        # Get frame IDs to be sampled
        global_frame_ids, sim_ids, local_frame_ids = self.determine_batch_frame_ids(idx)

        output = torch.full(
            (self.n_variants, self.batch_size, self.n_channels, *self.frame_size),
            float("nan"),
            dtype=torch.float32,
        )

        # Iterate over simulations involved and load all frames from each video at once
        for this_sim_id in np.unique(sim_ids):
            # mask: (batch_size,), dtype=bool
            mask_over_batch = sim_ids == this_sim_id
            # local_frame_ids_this_sim: (n_frames_this_sim,), dtype=int
            local_frame_ids_this_sim = local_frame_ids[mask_over_batch]

            # Load frames from all variants of this simulation
            sim_name = self.all_sim_names[this_sim_id]
            variant_paths = self.video_paths_by_sim_names[sim_name]
            for variant_id, video_path in enumerate(variant_paths):
                # Load frames from video
                # frames: list of (H, W, C=3) arrays
                frames, fps = read_frames_from_video(
                    video_path, frame_indices=local_frame_ids_this_sim.tolist()
                )

                # Apply necessary transforms
                # frames: (n_frames_this_sim, H, W, C=3), dtype=np.uint8
                frames = np.array(frames)
                assert frames.dtype == np.uint8, "Error: expected uint8 image array"
                # Convert image convention: transform [0, 255] uint8 numpy array to
                # [0, 1] float torch tensor and permute (N, H, W, C) to (N, C, H, W)
                frames = torch.from_numpy(frames)
                frames = (frames / 255.0).to(torch.float32)
                frames = frames.permute(0, 3, 1, 2)
                # Apply caller-specified transform
                if self.transform:
                    frames = self.transform(frames)
                # Check if n_channels is consistent with caller-specified value
                assert len(frames.shape) == 4 and frames.shape[1] == self.n_channels, (
                    f"Error: transform should return a tensor of shape "
                    f"(N, C, H, W) where C matches n_channels ({self.n_channels}), "
                    f"but got {frames.shape}"
                )

                # Populate output tensor
                output[variant_id, mask_over_batch, :, :, :] = frames

        # Check and return loaded data
        assert not torch.isnan(output).any(), (
            "Output contains NaN values; something went wrong in __getitem__ (some "
            "parts of the output tensor were not populated correctly)"
        )
        return output


def round_up_to_multiple(x: int, multiple_of: int) -> int:
    return ((x + multiple_of - 1) // multiple_of) * multiple_of


def atomic_batch_to_file(
    atomic_batch: np.ndarray, fps: int, output_path: Path, spacing: int = 10
):
    """Write an atomic batch of frames to a video file.

    Args:
        atomic_batch (np.ndarray): Array of shape (n_variants, n_frames,
            n_channels, height, width)
        output_path (Path): Path to save the video file.
        fps (int): Frames per second for the output video.
        spacing (int, optional): Number of pixels to insert between
    """
    n_variants, n_frames, n_channels, n_rows, n_cols = atomic_batch.shape
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
        for frame_idx in range(atomic_batch.shape[1]):
            image = np.zeros((n_rows_total, n_cols_total, n_channels), dtype=np.uint8)
            for variant_idx in range(atomic_batch.shape[0]):
                start_col = variant_idx * (n_cols + spacing)
                end_col = start_col + n_cols
                selection = atomic_batch[variant_idx, frame_idx, :, :, :]
                selection = selection.transpose(1, 2, 0)
                selection = (selection * 255).astype(np.uint8)
                image[:n_rows, start_col:end_col, :] = selection
            video_writer.append_data(image.squeeze())


def file_to_atomic_batch(
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
        (n_variants, n_frames, n_channels, n_rows, n_cols), dtype=np.uint8
    )
    for frame_idx, frame in enumerate(frames):
        for variant_idx in range(n_variants):
            start_col = variant_idx * (n_cols + spacing)
            end_col = start_col + n_cols
            selection = frame[:n_rows, start_col:end_col] / 255.0
            selection = selection.transpose(2, 0, 1)
            atomic_batch[variant_idx, frame_idx, :, :, :] = selection

    return atomic_batch
