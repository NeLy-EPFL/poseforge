from torch.utils.data import Dataset
from pathlib import Path

from biomechpose.pose_estimation.sampler import (
    load_atomic_batch_frames,
    load_atomic_batch_sim_data,
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
        frames = load_atomic_batch_frames(
            mp4_path,
            self.n_variants,
            self.image_size,
            self.n_channels,
            self.frames_serialization_spacing,
        )

        # Load labels data
        sim_data = load_atomic_batch_sim_data(h5_path, self.label_keys)

        return frames, sim_data


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
