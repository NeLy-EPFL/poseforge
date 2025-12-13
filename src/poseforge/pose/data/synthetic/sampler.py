import torch
import numpy as np
from collections import defaultdict
from typing import Callable

from .sim_data_seq import SimulatedDataSequence


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

        # Check if all simulations have the same metadata
        metadata_li = [seq.get_sim_data_metadata() for seq in simulated_data_sequences]
        md_ref = metadata_li[0]
        for metadata in metadata_li[1:]:
            assert metadata == md_ref, "All simulations must have the same metadata"
        self.label_keys = {
            "dof_angles": md_ref["dof_angles"]["keys"],
            "keypoint_pos": md_ref["keypoint_pos"]["keys"],
            "mesh_pos": md_ref["mesh_pose6d"]["keys"],
            "mesh_quat": md_ref["mesh_pose6d"]["keys"],
            "body_seg_maps": md_ref["segmentation_labels"]["keys"],
        }

        # Check number of variants per frame
        assert (
            len(set([seq.n_variants for seq in simulated_data_sequences])) == 1
        ), "All simulations must have the same number of variants"
        self.n_variants = simulated_data_sequences[0].n_variants

        # Check image size and FPS
        _image_sizes = set()
        _fps = set()
        for seq in simulated_data_sequences:
            _image_sizes.add(seq.frame_size)
            _fps.add(seq.fps)
        assert len(_image_sizes) == 1, "All simulations must have the same image size"
        assert len(_fps) == 1, "All simulations must have the same FPS"
        self.frame_size = _image_sizes.pop()
        self.fps = _fps.pop()

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
        frames = torch.from_numpy(frames).to(torch.float32) / 255.0
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
