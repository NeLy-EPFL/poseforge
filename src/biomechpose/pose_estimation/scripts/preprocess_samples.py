from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import trange
from time import time

from biomechpose.pose_estimation.sampler import (
    SimulatedDataSequence,
    SyntheticFramesSampler,
    save_atomic_batch_frames,
    save_atomic_batch_sim_data,
)


def extract_atomic_batches(
    atomic_batch_nframes: int,
    atomic_batch_nvariants_max: int,
    minimum_time_diff_frames: int,
    input_basedir: Path | str,
    nmf_sim_rendering_basedir: Path | str,
    output_dir: Path | str,
    n_jobs: int = -1,
    logging_interval: int = 100,
    cache_metadata: bool = True,
    use_cached_metadata: bool = True,
    video_spacing: int = 10,
) -> None:
    """Extract atomic batches from simulated data. This makes data loading
    faster during pose estimation training.

    Each "atomic batch" contains a small number of frames translated into
    the domain of experimental data using a number of different style
    transfer models (each referred to as a "variant"). Each batch therefore
    contains:
        - Simulated frames: n_variants * n_frames, saved as a video of
          n_frames frames. In each frame of the video, the n_variants
          variants are tiled horizontally with a fixed spacing of
          `video_spacing` pixels between each variant.
        - Simulated data: a single H5 file containing the simulated data
          (3D joint positions, etc.) for these frames (note that different
          variants share the same simulated data since they come from the
          same NeuroMechFly simulation).

    Args:
        atomic_batch_nframes (int): The number of frames in each atomic
            batch.
        atomic_batch_nvariants_max (int): The maximum number of variants to
            include in each atomic batch.
        minimum_time_diff_frames (int): The minimum time difference between
            frames for them to be considered different (and therefore
            "randomly" sampled).
        input_basedir (Path | str): The base directory containing synthetic
            videos.
        nmf_sim_rendering_basedir (Path | str): The base directory for NMF
            simulation rendering.
        output_dir (Path | str): The output directory for atomic batches.
        n_jobs (int, optional): The number of jobs to run in parallel. 0 =
            run in a plain loop; -1 = use all available cores. Defaults to
            -1.
        logging_interval (int, optional): The interval at which to log
            progress. Defaults to 100.
        cache_metadata (bool, optional): Whether to cache metadata when
            indexing simulations info (e.g. length of simulation). Defaults
            to True.
        use_cached_metadata (bool, optional): Whether to use cached
            metadata if they are available. Defaults to True.
        video_spacing (int, optional): The spacing (in pixels) between
            variants in the output video. Defaults to 10.
    """
    input_basedir = Path(input_basedir)
    nmf_sim_rendering_basedir = Path(nmf_sim_rendering_basedir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set the following to a non-None value to limit the scope of the job for testing
    max_n_simulations: int | None = None
    max_n_variants: int | None = None

    # Define all synthetic videos
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
    print("Indexing all simulation data in parallel. This may take a few minutes.")

    def build_sim_data_seq(sim_path):
        synthetic_video_paths = [sim_path / f"{model}.mp4" for model in models]
        exp_trial, segment, subsegment = synthetic_video_paths[0].parent.parts[-3:]
        simulated_labels_path = (
            nmf_sim_rendering_basedir
            / exp_trial
            / segment
            / subsegment
            / "processed_simulation_data.h5"
        )
        return SimulatedDataSequence(
            synthetic_video_paths,
            simulated_labels_path,
            sim_name=f"{exp_trial}/{segment}/{subsegment}",
            cache_metadata=cache_metadata,
            use_cached_metadata=use_cached_metadata,
        )

    simulated_data_sequences = Parallel(n_jobs=-1)(
        delayed(build_sim_data_seq)(sim_path)
        for sim_path in tqdm(input_simulation_paths, disable=None)
    )
    print(f"Indexed {len(simulated_data_sequences)} simulations")

    # Create sampler
    sampler = SyntheticFramesSampler(
        simulated_data_sequences,
        batch_size=atomic_batch_nframes,
        sampling_stride=minimum_time_diff_frames,
        load_labels=True,
    )
    print(
        f"Dataset initialized with {len(sampler)} batches of samples from "
        f"{len(sampler.simulated_data_sequences)} simulations. "
        f"Total number of usable frames: {sampler.total_n_frames} "
        f"({sampler.total_n_frames / 300:.2f} seconds at 300 FPS)."
    )

    # Figure out how many atomic batches we need to split each batch into
    n_atomic_batches_per_batch = sampler.n_variants // atomic_batch_nvariants_max
    if sampler.n_variants % atomic_batch_nvariants_max != 0:
        raise ValueError(
            f"atomic_batch_nvariants_max ({atomic_batch_nvariants_max}) must be a "
            f"divisor of the total number of variants ({sampler.n_variants})"
        )
    print(
        f"Each batch will be split into {n_atomic_batches_per_batch} atomic batches "
        f"of {atomic_batch_nvariants_max} variants each"
    )

    # Define payload
    start_time = time()

    def process_batch(batch_idx: int):
        frames, labels = sampler[batch_idx]
        # Split into multiple atomic batches if n_variants > atomic_batch_nvariants_max
        for atomic_batch_group_idx in range(n_atomic_batches_per_batch):
            variant_idx_start = atomic_batch_group_idx * atomic_batch_nvariants_max
            variant_idx_end = (atomic_batch_group_idx + 1) * atomic_batch_nvariants_max
            atomic_batch_frames = frames[variant_idx_start:variant_idx_end, :, :, :, :]

            global_atomic_batch_idx = (
                batch_idx * n_atomic_batches_per_batch + atomic_batch_group_idx
            )
            filename_stem = f"atomicbatch{global_atomic_batch_idx:08d}_batch{batch_idx:05d}_variantsgroup{variant_idx_start:02d}"
            save_atomic_batch_frames(
                atomic_batch_frames,
                output_dir / f"{filename_stem}_frames.mp4",
                fps=sampler.fps,
                spacing=video_spacing,
            )
            save_atomic_batch_sim_data(
                labels, output_dir / f"{filename_stem}_labels.h5"
            )
        if batch_idx % logging_interval == 0:
            elapsed = time() - start_time
            print(
                f"Processed batch {batch_idx}/{len(sampler)} "
                f"({(batch_idx + 1) / len(sampler) * 100:.2f}%) "
                f"in {elapsed:.1f} seconds"
            )

    # Execute jobs
    if n_jobs == 0:
        print("Starting to process and save all batches in series")
        for batch_idx in trange(len(sampler)):
            process_batch(batch_idx)
    else:
        print(f"Starting to process and save all batches in parallel")
        parallel_executor = Parallel(n_jobs=n_jobs)
        effective_n_jobs = parallel_executor._effective_n_jobs()
        print(f"User specified n_jobs={n_jobs}; this is in effect {effective_n_jobs}")
        parallel_executor(
            delayed(process_batch)(i_batch)
            for i_batch in trange(len(sampler), disable=None)
        )
    print("All done")


if __name__ == "__main__":
    # CLI
    import tyro

    tyro.cli(extract_atomic_batches)
    # CLI example:
    # python src/biomechpose/pose_estimation/scripts/preprocess_samples.py \
    #     --atomic-batch-nframes 32 \
    #     --atomic-batch-nvariants-max 4 \
    #     --minimum-time-diff-frames 60 \
    #     --input-basedir bulk_data/style_transfer/production/translated_videos/BO_Gal4_fly1_trial001 \
    #     --nmf-sim-rendering-basedir bulk_data/nmf_rendering \
    #     --output-dir bulk_data/pose_estimation/atomic_batches

    # # Example usage not via CLI
    # extract_atomic_batches(
    #     atomic_batch_nframes=32,
    #     atomic_batch_nvariants_max=4,
    #     minimum_time_diff_frames=60,
    #     input_basedir="bulk_data/style_transfer/production/translated_videos",
    #     nmf_sim_rendering_basedir="bulk_data/nmf_rendering/",
    #     output_dir="bulk_data/pose_estimation/atomic_batches",
    #     n_jobs=-1,
    #     logging_interval=100,
    #     cache_metadata=True,
    #     use_cached_metadata=True,
    #     video_spacing=10,
    # )
