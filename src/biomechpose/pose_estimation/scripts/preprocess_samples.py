import numpy as np
import pandas as pd
import imageio.v2 as imageio
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from skimage.transform import resize
from pathlib import Path
from tqdm import trange
from torchvision.transforms import Grayscale

from biomechpose.pose_estimation.sampler import (
    SyntheticFramesSampler,
    atomic_batch_to_file,
)
from biomechpose.simulate_nmf.postprocessing import SegmentLabelParser
from biomechpose.util import (
    read_frames_from_video,
    default_video_writing_ffmpeg_params,
    df_contains_expanded_columns,
    expand_ndarray_dataframe_columns,
    restore_expandex_dataframe_columns,
)


def extract_synthetic_frames_for_atomic_batches(
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
            / f"synth_videos_batchsample{sample_idx:06d}_variantgroup{variant_group_idx:02d}.mp4"
        )
        atomic_batch_to_file(atomic_batch, fps, output_path)


def extract_simulated_seg_labels_for_atomic_batches(
    sampler: SyntheticFramesSampler,
    sample_idx: int,
    segmentation_label_parser: SegmentLabelParser,
    nmf_sim_rendering_basedir: Path,
    output_dir: Path,
    n_rendering_colorcodings: int = 2,
):
    _, sim_ids, local_frame_ids = sampler.determine_batch_frame_ids(sample_idx)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract labels for each simulation involved in the batch
    labels_all = np.zeros((sampler.batch_size, *sampler.frame_size), dtype=np.uint8)
    for sim_id in np.unique(sim_ids):
        sim_name = sampler.all_sim_names[sim_id]
        mask = sim_ids == sim_id
        num_frames_in_sim = mask.sum()

        segmentation_labels_path = (
            nmf_sim_rendering_basedir / sim_name / "segmentation_labels.npz"
        )
        segmentation_labels_data = np.load(segmentation_labels_path)
        sim_length = segmentation_labels_data["labels"].shape[0]
        mask_within_sim = np.isin(
            np.arange(sim_length), local_frame_ids[sim_ids == sim_id]
        )
        labels = segmentation_labels_data["labels"][mask_within_sim, :, :]
        labels = labels.astype(np.uint8)
        label_keys = segmentation_labels_data["label_keys"]
        assert len(label_keys) <= 255, "Too many unique labels (>255) to fit in uint8"
        labels_resampled = (
            F.interpolate(
                torch.from_numpy(labels)[:, None, :, :],  # additional dim for channels
                size=sampler.frame_size,
                mode="nearest",
            )
            .squeeze(1)
            .numpy()
        )  # remove channels dim
        labels_all[mask, :, :] = labels_resampled

    # Save label images as a video
    output_path = output_dir / f"seg_labels_batchsample{sample_idx:06d}.npz"
    np.savez_compressed(output_path, labels=labels_all, label_keys=label_keys)
    np.savez(str(output_path) + ".cpr", labels=labels_all, label_keys=label_keys)


def extract_simulated_kinematics_labels_for_atomic_batches(
    sampler: SyntheticFramesSampler,
    sample_idx: int,
    nmf_sim_rendering_basedir: Path,
    output_dir: Path,
):
    import time

    st = time.time()
    _, sim_ids, local_frame_ids = sampler.determine_batch_frame_ids(sample_idx)
    print(f"  kinstates: sampling done after {(time.time()-st)*1000:.1f} ms")
    assert np.diff(sim_ids).min() >= 0, "sim_ids should be monotonically increasing"

    output_dir.mkdir(parents=True, exist_ok=True)

    kinematics_rows_all = []
    for sim_id in np.unique(sim_ids):
        sim_name = sampler.all_sim_names[sim_id]
        kinematics_path = (
            nmf_sim_rendering_basedir / sim_name / "processed_kinematic_states.pkl"
        )
        print(
            f"  kinstates: loading {kinematics_path} after {(time.time()-st)*1000:.1f} ms"
        )
        kinematics_df = pd.read_pickle(kinematics_path)
        if df_contains_expanded_columns(kinematics_df):
            kinematics_df = restore_expandex_dataframe_columns(kinematics_df)
        print(f"  kinstates: loading done after {(time.time()-st)*1000:.1f} ms")
        mask_within_sim = np.isin(
            np.arange(len(kinematics_df)), local_frame_ids[sim_ids == sim_id]
        )
        kinematics_df = kinematics_df.loc[mask_within_sim, :]
        kinematics_rows_all.append(kinematics_df)
    kinematics_df_all = pd.concat(kinematics_rows_all, axis=0).reset_index(drop=True)
    assert (
        len(kinematics_df_all) == sampler.batch_size
    ), f"Expected {sampler.batch_size} rows, got {len(kinematics_df_all)}"
    print(f"  kinstates: formatting done after {(time.time()-st)*1000:.1f} ms")
    output_path = output_dir / f"kinematics_labels_batchsample{sample_idx:06d}.pkl"
    kinematics_df_all_expanded = expand_ndarray_dataframe_columns(kinematics_df_all)
    kinematics_df_all_expanded.to_pickle(output_path)
    print(f"  kinstates: saved to {output_path}")


if __name__ == "__main__":
    # Parameters
    atomic_batch_nframes = 32
    atomic_batch_nvariants_max = 4
    transform = Grayscale(num_output_channels=1)
    minimum_time_diff_frames = 60  # 0.2s at 300Hz
    nmf_sim_rendering_basedir = Path("bulk_data/nmf_rendering/")
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

    # Create segmentation label parser
    segmentation_label_parser = SegmentLabelParser()

    # Define processing payload
    def process_one_batch(batch_id: int):
        import time

        st = time.time()
        extract_synthetic_frames_for_atomic_batches(
            sampler, batch_id, atomic_batch_nvariants_max, output_dir, sampler.fps
        )
        print(
            f"Extracted frames for batch {batch_id} in {(time.time()-st)*1000:.1f} ms"
        )

        st = time.time()
        extract_simulated_seg_labels_for_atomic_batches(
            sampler,
            batch_id,
            segmentation_label_parser,
            nmf_sim_rendering_basedir,
            output_dir,
        )
        print(
            f"Extracted seg labels for batch {batch_id} in {(time.time()-st)*1000:.1f} ms"
        )

        st = time.time()
        extract_simulated_kinematics_labels_for_atomic_batches(
            sampler, batch_id, nmf_sim_rendering_basedir, output_dir
        )
        print(
            f"Extracted kinematics labels for batch {batch_id} in {(time.time()-st)*1000:.1f} ms"
        )

    # Sequential processing
    for batch_id in trange(len(sampler), total=len(sampler)):
        process_one_batch(batch_id)
