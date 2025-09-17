import numpy as np
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import trange
from torchvision.transforms import Grayscale

from biomechpose.pose_estimation.sampler import (
    SyntheticFramesSampler,
    atomic_batch_to_file,
)


def extract_atomic_batches(
    sampler: SyntheticFramesSampler,
    sample_idx: int,
    n_variants_per_atomic_batch: int,
    output_dir: Path,
    fps: int = 1,
):
    batch = sampler[sample_idx].numpy()
    assert batch.dtype == np.float32, "Expected float32 input"
    assert batch.min() >= 0 and batch.max() <= 1, "Expected input in [0, 1]"
    n_variants, n_frames, n_channels, height, width = batch.shape
    assert n_channels == 1, "Expected grayscale input"
    n_variant_groups = (
        n_variants + n_variants_per_atomic_batch - 1
    ) // n_variants_per_atomic_batch
    for variant_group_idx in range(n_variant_groups):
        start_variant = variant_group_idx * n_variants_per_atomic_batch
        end_variant = min(start_variant + n_variants_per_atomic_batch, n_variants)
        atomic_batch = batch[start_variant:end_variant, :, :, :, :]
        output_path = (
            output_dir
            / f"sample{sample_idx:06d}_variantgroup{variant_group_idx:02d}.mp4"
        )
        atomic_batch_to_file(atomic_batch, fps, output_path)


if __name__ == "__main__":
    # Parameters
    atomic_batch_nframes = 32
    atomic_batch_nvariants_max = 4
    transform = Grayscale(num_output_channels=1)
    minimum_time_diff_frames = 60  # 0.2s at 300Hz
    output_dir = Path("bulk_data/pose_estimation/atomic_batches")

    # Set the following to a non-None value to limit the scope of the job for testing
    max_n_simulations: int | None = None
    max_n_variants: int | None = None

    # Define all synthetic videos
    input_basedir = Path("bulk_data/style_transfer/production/translated_videos")
    assert input_basedir.is_dir(), f"`input_basedir` {input_basedir} is not a directory"
    input_simulation_paths = sorted(list(input_basedir.rglob("subsegment_*")))
    if max_n_simulations is not None:
        input_simulation_paths = input_simulation_paths[:max_n_simulations]
    print(f"Found {len(input_simulation_paths)} simulations to use for pretraining")

    # Define image variants (i.e. style transfer output by different models)
    models = sorted(
        [path.stem for path in input_simulation_paths[0].glob("translated_*.mp4")]
    )
    if max_n_variants is not None:
        models = models[:max_n_variants]
    n_variants = len(models)
    print(f"Using {n_variants} input variants")

    # Index all video paths
    video_paths_by_sim_names = {}
    for sim_path in input_simulation_paths:
        sim_name = "/".join(sim_path.parts[-3:])
        video_paths_by_sim_names[sim_name] = [
            sim_path / f"{model}.mp4" for model in models
        ]

    # Create dataset
    sampler = SyntheticFramesSampler(
        video_paths_by_sim_names,
        batch_size=atomic_batch_nframes,
        sampling_stride=minimum_time_diff_frames,
        transform=transform,
        n_channels=1,
    )
    print(
        f"Dataset initialized with {len(sampler)} batches of samples from "
        f"{len(sampler.all_sim_names)} simulations. Total number of "
        f"usable frames: {sampler.total_n_frames} "
        f"({sampler.total_n_frames / 300:.2f} seconds at 300 FPS)."
    )

    n_atomic_batches_across_variants = (
        n_variants + atomic_batch_nvariants_max - 1
    ) // atomic_batch_nvariants_max
    n_atomic_batches_across_sims = (
        sampler.total_n_frames + atomic_batch_nframes - 1
    ) // atomic_batch_nframes

    # # Sequential processing
    # for batch_id in trange(len(sampler), total=len(sampler)):
    #     extract_atomic_batches(
    #         sampler, batch_id, atomic_batch_nvariants_max, output_dir, sampler.fps
    #     )

    # Parallel processing
    n_jobs = -1  # Use all available CPU cores
    Parallel(n_jobs=n_jobs)(
        delayed(extract_atomic_batches)(
            sampler, batch_id, atomic_batch_nvariants_max, output_dir, sampler.fps
        )
        for batch_id in trange(len(sampler), total=len(sampler))
    )
