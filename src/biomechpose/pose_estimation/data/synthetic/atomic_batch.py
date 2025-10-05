import torch
import numpy as np
import h5py
import imageio.v2 as imageio
import logging
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from biomechpose.util import (
    read_frames_from_video,
    round_up_to_multiple,
    default_video_writing_ffmpeg_params,
    get_hardware_availability,
)


class AtomicBatchDataset(Dataset):
    def __init__(
        self,
        data_dirs: list[Path],
        n_variants: int,
        image_size: tuple[int, int],
        n_channels: int = 1,
        frames_serialization_spacing: int = 10,
        load_dof_angles: bool = False,
        load_keypoint_positions: bool = False,
        load_body_segment_maps: bool = False,
    ):
        # Find all .h5 and .mp4 files in the provided directories
        all_h5_files = set()
        all_mp4_files = set()
        for data_dir in data_dirs:
            if not data_dir.is_dir():
                raise ValueError(f"Provided path {data_dir} is not a directory.")
            mp4_files = list(data_dir.rglob("atomicbatch*_frames.mp4"))
            h5_files = list(data_dir.rglob("atomicbatch*_labels.h5"))
            all_mp4_files.update(mp4_files)
            all_h5_files.update(h5_files)

        # Ensure that the number of .h5 and .mp4 files match
        mp4_paths_lookup = {
            str(path).replace("_frames.mp4", ""): path for path in all_mp4_files
        }
        h5_paths_lookup = {
            str(path).replace("_labels.h5", ""): path for path in all_h5_files
        }
        if len(mp4_paths_lookup) != len(all_mp4_files):
            raise ValueError("Duplicate .mp4 basenames found.")
        if len(h5_paths_lookup) != len(all_h5_files):
            raise ValueError("Duplicate .h5 basenames found.")
        if set(mp4_paths_lookup.keys()) != set(h5_paths_lookup.keys()):
            raise ValueError(
                f"Mismatch between .mp4 files ({len(all_mp4_files)} found) and "
                f".h5 files ({len(all_h5_files)} found)."
            )
        if len(all_mp4_files) == 0:
            raise ValueError("No atomic batches found in the provided directories.")

        # Sort the basenames to ensure consistent ordering
        self.atomic_batch_names = sorted(list(mp4_paths_lookup.keys()))
        self.atomic_batches = [
            (mp4_paths_lookup[basename], h5_paths_lookup[basename])
            for basename in self.atomic_batch_names
        ]

        # Save other attributes
        self.n_variants = n_variants
        self.image_size = image_size
        self.n_channels = n_channels
        self.frames_serialization_spacing = frames_serialization_spacing
        self.label_keys = []
        if load_dof_angles:
            self.label_keys.append("dof_angles")
        if load_keypoint_positions:
            self.label_keys.append("keypoint_pos")
        if load_body_segment_maps:
            self.label_keys.append("body_seg_maps")

    def __len__(self):
        return len(self.atomic_batches)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range.")
        mp4_path, h5_path = self.atomic_batches[idx]

        # Load frames data
        frames = self.load_atomic_batch_frames(
            mp4_path,
            self.n_variants,
            self.image_size,
            self.n_channels,
            self.frames_serialization_spacing,
        )

        # Load labels data
        sim_data = self.load_atomic_batch_sim_data(h5_path, self.label_keys)

        return frames, sim_data

    @staticmethod
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
        atomic_batch = atomic_batch.detach().cpu().numpy()
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
                image = np.zeros(
                    (n_rows_total, n_cols_total, n_channels), dtype=np.uint8
                )
                for variant_idx in range(atomic_batch.shape[0]):
                    start_col = variant_idx * (n_cols + spacing)
                    end_col = start_col + n_cols
                    selection = atomic_batch[variant_idx, frame_idx, :, :, :]
                    # Move channel to the last dimension
                    selection = selection.transpose(1, 2, 0)
                    selection = (selection * 255).astype(np.uint8)
                    image[:n_rows, start_col:end_col, :] = selection
                video_writer.append_data(image.squeeze())

    @staticmethod
    def load_atomic_batch_frames(
        video_path: Path,
        n_variants: int,
        image_size: tuple[int, int],
        n_channels: int = 1,
        spacing: int = 10,
    ) -> torch.Tensor:
        """Read an atomic batch of frames from a video file.

        Args:
            video_path (Path): Path to the video file.
            n_variants (int): Number of variants in the atomic batch.
            image_size (tuple[int, int]): (height, width) of each variant.
            n_channels (int, optional): Number of image channels. Defaults to 1.
            spacing (int, optional): Number of pixels inserted between
                variants. Defaults to 10.

        Returns:
            atomic_batch (torch.Tensor): Tensor of shape (n_variants,
                n_frames, n_channels, height, width)
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

    @staticmethod
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

    @staticmethod
    def load_atomic_batch_sim_data(
        input_path: Path, keys: list[str] | None
    ) -> dict[str, torch.Tensor]:
        """Load simulation data for an atomic batch from an HDF5 file.

        Args:
            input_path (Path): Path to the HDF5 file.
            keys (list[str] | None): List of keys to load. If None, all keys are loaded.

        Returns:
            dict[str, torch.Tensor]: Dictionary of simulation data.
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


def concat_atomic_batches(
    frames: torch.Tensor, sim_data: dict[str, torch.Tensor] | None = None
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Concatenate a group of atomic batches along the batch (n_frames)
    dimension.

    Args:
        frames (torch.Tensor): Tensor of shape
            (n_atomic_batches, n_variants, n_frames, n_channels, height, width)
        sim_data (dict[str, torch.Tensor], optional): Optional dictionary
            of simulation data, where each value is a tensor of shape
            (n_atomic_batches, n_frames, ...). Note that there is no
            n_variants dimension here because all variants come from the
            same simulation. If supplied, each dict key-value pair will be
            repeated along the n_variants dimension and then concatenated
            along the n_frames dimension so as to match the concatenated
            atomic_batches. Defaults to None.

    Returns:
        If sim_data is None:
            torch.Tensor: Concatenated atomic batch of shape
                (n_variants, n_atomic_batches * n_frames, n_channels, height, width)
        If sim_data is not None:
            torch.Tensor: Concatenated atomic batch of shape
                (n_variants, n_atomic_batches * n_frames, n_channels, height, width)
            dict[str, torch.Tensor]: Dictionary of concatenated simulation
                data, where each value is a tensor of shape
                (n_atomic_batches * n_frames, ...)
    """
    n_atomic_batches, n_variants, _, _, _, _ = frames.shape
    # list_of_atomic_batches: each element has shape (n_variants, n_frames, n_channels
    # n_rows, n_cols). Then concatenate along dim 1 (n_frames)
    list_of_atomic_batches = [frames[i, ...] for i in range(n_atomic_batches)]
    concatenated_batch = torch.cat(list_of_atomic_batches, dim=1)
    concatenated_batch = concatenated_batch.to(frames.device)

    if sim_data is None:
        return concatenated_batch
    else:
        concatenated_sim_data = {}
        for key, data in sim_data.items():
            # data has shape (n_atomic_batches, n_frames, ...). Now add a n_variants dim
            data = data.unsqueeze(1)
            # Repeat along n_variants times along the new dim
            repeats = [1] * data.ndim
            repeats[1] = n_variants
            data = data.repeat(*repeats)
            # list_of_sim_data: each element has shape (n_variants, n_frames, ...).
            list_of_sim_data = [data[i, ...] for i in range(n_atomic_batches)]
            # Then concatenate along dim 1 (n_frames)
            concatenated_data = torch.cat(list_of_sim_data, dim=1)
            concatenated_data = concatenated_data.to(frames.device)
            concatenated_sim_data[key] = concatenated_data
        return concatenated_batch, concatenated_sim_data


def collapse_batch(
    frames: torch.Tensor, sim_data: dict[str, torch.Tensor] | None = None
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reshape a group of atomic batches into a single batch, and
    collapse the n_variants and n_samples dimensions.

    Args:
        frames (torch.Tensor): Tensor of shape
            (n_variants, n_atomic_batches * n_samples, C, H, W)
        sim_data (dict[str, torch.Tensor], optional): Optional dictionary
            of simulation data, already made consistent with the frames
            tensor using concat_atomic_batches(). Each value should be a
            tensor of shape (n_atomic_batches * n_samples, ...).

    Returns:
        If sim_data is None:
            torch.Tensor: Collapsed batch of shape
                (n_variants * n_atomic_batches * n_samples, C, H, W).
        If sim_data is not None:
            torch.Tensor: Collapsed batch of shape
                (n_variants * n_atomic_batches * n_samples, C, H, W).
            dict[str, torch.Tensor]: Dictionary of simulation data, where
                each value is a tensor of shape
                (n_variants * n_atomic_batches * n_samples, ...).
    """
    n_variants, n_samples_all_atomic_batches, n_channels, nrows, ncols = frames.shape
    # collapsed_frames:
    # (n_variants * n_atomic_batches * atomic_batch_n_samples, C, H, W)
    collapsed_frames = frames.view(
        n_variants * n_samples_all_atomic_batches, n_channels, nrows, ncols
    )

    if sim_data is None:
        return collapsed_frames
    else:
        collapsed_sim_data = {}
        for key, data in sim_data.items():
            # data has shape (n_variants, n_atomic_batches * n_samples, ...).
            collapsed_data = data.view(
                n_variants * n_samples_all_atomic_batches, *data.shape[2:]
            )
            collapsed_sim_data[key] = collapsed_data
        return collapsed_frames, collapsed_sim_data


def init_atomic_dataset_and_dataloader(
    data_dirs: list[str | Path],
    atomic_batch_n_samples: int,
    atomic_batch_n_variants: int,
    image_size: tuple[int, int],
    batch_size: int,
    load_dof_angles: bool = False,
    load_keypoint_positions: bool = False,
    load_body_segment_maps: bool = False,
    shuffle: bool = False,
    num_workers: int | None = None,
    num_channels: int = 3,
    pin_memory: bool = True,
    drop_last: bool = True,
):
    """
    Initializes an AtomicBatchDataset and a corresponding DataLoader for
    training.

    Args:
        data_dirs (list[str | Path]): List of directories that (potentially
            recursively) contain the data.
        atomic_batch_n_samples (int): Number of samples (unique frames from
            NeuroMechFly simulation) in each atomic batch.
        atomic_batch_n_variants (int): Number of variants in each atomic
            batch.
        image_size (tuple[int, int]): Size of the images (height, width).
        batch_size (int): Desired batch size. Must be a multiple of
            atomic_batch_n_samples.  TODO: further divide by n_variants?
        load_dof_angles (bool, optional): Whether to load degree-of-freedom
            angles. Defaults to False.
        load_keypoint_positions (bool, optional): Whether to load keypoint
            positions. Defaults to False.
        load_body_segment_maps (bool, optional): Whether to load body
            segment maps. Defaults to False.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to
            False.
        num_workers (int | None, optional): Number of worker threads for
            data loading. If None, uses available CPU cores.
        num_channels (int, optional): Number of image channels. Default 3.
        pin_memory (bool, optional): Whether to use pinned memory in
            DataLoader. Defaults to True.
        drop_last (bool, optional): Whether to drop the last incomplete
            batch. Defaults to True.

    Returns:
        dataset (AtomicBatchDataset):
            The initialized dataset.
        dataloader (torch.utils.data.DataLoader):
            The corresponding dataloader.
    """
    # Set up atomic datasets
    dataset = AtomicBatchDataset(
        data_dirs=[Path(path) for path in data_dirs],
        n_variants=atomic_batch_n_variants,
        image_size=image_size,
        n_channels=num_channels,
        load_dof_angles=load_dof_angles,
        load_keypoint_positions=load_keypoint_positions,
        load_body_segment_maps=load_body_segment_maps,
    )

    # Check if batch size is valid
    n_atomic_batches_per_batch = batch_size // atomic_batch_n_samples
    if not batch_size % atomic_batch_n_samples == 0:
        raise ValueError(
            "`train_batch_size` must be a multiple of `atomic_batch_n_samples`"
        )

    # Create parallel dataloaders
    num_workers = num_workers
    if num_workers is None:
        hardware_avail = get_hardware_availability()
        num_workers = hardware_avail["num_cpu_cores_available"]
        logging.info(f"Using {num_workers} data loading workers")
    dataloader = DataLoader(
        dataset,
        batch_size=n_atomic_batches_per_batch,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataset, dataloader


def _test_throughput():
    from time import time
    from torch.utils.data import DataLoader

    data_dirs = list(
        Path("bulk_data/pose_estimation/atomic_batches/").glob("BO_Gal4_*")
    )
    dataset = AtomicBatchDataset(
        data_dirs=data_dirs,
        n_variants=4,
        image_size=(256, 256),
        n_channels=1,
        frames_serialization_spacing=10,
        load_dof_angles=True,
        load_keypoint_positions=True,
        load_body_segment_maps=True,
    )
    print(f"Dataset length: {len(dataset)}")

    print("***** Test loading throughput using a single core *****")
    n_samples = 30
    st = time()
    for i in range(n_samples):
        sample = dataset[i]
    walltime = time() - st
    samples_per_second = n_samples / walltime
    proj_total_time = len(dataset) / samples_per_second
    print(f"Loading throughput: {samples_per_second:.2f} samples/second")
    print(f"Projected time to iterate over dataset: {proj_total_time:.2f} seconds")

    print("***** Test loading with a 10-worker DataLoader with batch_size=10 *****")
    dataloader = DataLoader(dataset, batch_size=10, num_workers=10)
    st = time()
    n_batches = 100
    total_samples = 0
    for i, (frames, labels) in enumerate(dataloader):
        total_samples += frames.shape[0]
        if i == n_batches:
            break
    walltime = time() - st
    samples_per_second = total_samples / walltime
    proj_total_time = len(dataset) / samples_per_second
    print(f"Loading throughput: {samples_per_second:.2f} samples/second")
    print(f"Projected time to iterate over dataset: {proj_total_time:.2f} seconds")


# if __name__ == "__main__":
#     _test_throughput()
