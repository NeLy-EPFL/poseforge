"""Extract a CUT training dataset with multiple geometric prior buffers.

This is the multi-prior successor to ``extract_dataset.py``. The original script
wrote a single domain-A image per frame (the grayscale NeuroMechFly render) plus
a binary silhouette mask. The sim-to-real pipeline now also renders, for every
simulated frame and pixel-aligned with the grayscale render, a set of geometric
prior buffers:

- ``processed_nmf_sim_render_shading.mp4``  -- uniform-albedo shape-from-shading
- ``processed_nmf_sim_depth.mp4``           -- colorized metric depth
- ``processed_nmf_sim_normals.mp4``         -- back-projected surface normals
- ``processed_nmf_sim_segment_id.mp4``      -- body-part segmentation colors

All of these go through the exact same crop/rotate in
``poseforge.neuromechfly.postprocessing`` as the grayscale render, so a given
``(subsegment, frame_index)`` selects every buffer at once and they stay
pixel-aligned.

What this script produces (CUT-style layout under ``output_dir``)::

    trainA/                 <- primary domain-A image (grayscale render), .jpg
    trainA_shading/         <- shading prior,       same stem, .png
    trainA_depth/           <- depth prior,         same stem, .png
    trainA_normals/         <- normals prior,       same stem, .png
    trainA_segmentation/    <- segmentation prior,  same stem, .png
    trainA_mask/            <- binary silhouette,   same stem, .png
    trainB/                 <- domain-B real Spotlight images, .jpg
    testA/  testA_*/  testB/            (validation split)

Every ``trainA*`` folder is written with identical file stems, so the i-th entry
of each folder corresponds to the same frame. Extraction is all-or-nothing per
frame: if any one buffer (or the mask) cannot be produced for a frame, none of
the buffers are written for that frame, preserving the cross-folder alignment
that CUT's aligned loaders rely on.

Note on domain B: the real Spotlight recordings have no synthetic geometry, so
only ``trainB`` / ``testB`` images are written -- there are no B-side priors.
How the A-side priors are actually consumed by CUT (extra input channels,
separate encoders, conditioning) is the second design axis and is handled on the
CUT side, not here.
"""

import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import tyro
from pvio.io import check_num_frames, read_frames_from_video
from tqdm import tqdm

# The spotlight (domain B) side is unchanged, so reuse the existing helpers.
try:
    from poseforge.style_transfer.scripts.extract_dataset import (
        extract_spotlight_recording_frames_from_specs,
        list_spotlight_recordings_and_num_frames,
    )
except ModuleNotFoundError:
    # Allows running this file directly as a script from this directory.
    from extract_dataset import (
        extract_spotlight_recording_frames_from_specs,
        list_spotlight_recordings_and_num_frames,
    )


# Key used for the primary domain-A image. This buffer is written to the bare
# "<split>A" folder; every other buffer goes to "<split>A_<key>".
PRIMARY_BUFFER_KEY = "image"

# Mapping from output buffer key -> rendered video filename inside each NMF
# subsegment directory. The primary "image" entry is the grayscale render that
# the previous dataset used; the rest are the geometric priors. Edit this dict
# (or override it on the command line) to add/remove priors.
DEFAULT_SIMULATION_BUFFER_VIDEOS: dict[str, str] = {
    PRIMARY_BUFFER_KEY: "processed_nmf_sim_render_grayscale.mp4",
    "shading": "processed_nmf_sim_render_shading.mp4",
    "depth": "processed_nmf_sim_depth.mp4",
    "normals": "processed_nmf_sim_normals.mp4",
    "segmentation": "processed_nmf_sim_segment_id.mp4",
}


def _domain_a_output_dir(output_dir: Path, split_prefix: str, buffer_key: str) -> Path:
    """Folder that holds one buffer for one split on the domain-A side.

    The primary image lands in "<split>A"; every prior lands in
    "<split>A_<buffer_key>"; the silhouette mask lands in "<split>A_mask".
    """
    if buffer_key == PRIMARY_BUFFER_KEY:
        return output_dir / f"{split_prefix}A"
    return output_dir / f"{split_prefix}A_{buffer_key}"


def _count_usable_nmf_frames(
    subsegment_dir: Path, buffer_filenames: dict[str, str]
) -> tuple[Path, int, str]:
    """Return (subsegment_dir, num_usable_frames, status_tag) for one subsegment.

    A subsegment is "usable" only if it has EVERY requested buffer video AND a
    matching ``postprocessed/segmentation_labels`` dataset (used both as the
    silhouette mask and as the frame-count ceiling). num_usable_frames is capped
    to the shortest of all these arrays so we never index past any of them.
    Failures return 0 frames with a tag; the caller logs and drops the
    subsegment so the random pool only ever contains fully-aligned frames.
    """
    frame_counts: list[int] = []
    for buffer_key, filename in buffer_filenames.items():
        video_path = subsegment_dir / filename
        if not video_path.is_file():
            return (subsegment_dir, 0, f"missing buffer video '{filename}' ({buffer_key})")
        try:
            frame_counts.append(check_num_frames(video_path))
        except Exception as exc:  # noqa: BLE001
            return (subsegment_dir, 0, f"failed to read buffer video '{filename}': {exc}")

    h5_path = subsegment_dir / "processed_simulation_data.h5"
    if not h5_path.is_file():
        return (subsegment_dir, 0, "missing simulation h5 (no mask available)")
    try:
        with h5py.File(h5_path, "r") as h5_file:
            if "postprocessed/segmentation_labels" not in h5_file:
                return (subsegment_dir, 0, "h5 has no postprocessed/segmentation_labels")
            num_segmentation_frames = int(
                h5_file["postprocessed/segmentation_labels"].shape[0]
            )
    except (OSError, KeyError) as exc:
        return (subsegment_dir, 0, f"failed to read h5: {exc}")
    frame_counts.append(num_segmentation_frames)

    num_usable = min(frame_counts)
    if num_usable <= 0:
        return (subsegment_dir, 0, "no usable frames")
    if len(set(frame_counts)) != 1:
        status = f"capped to {num_usable} (buffer/mask frame counts differ: {frame_counts})"
    else:
        status = "ok"
    return (subsegment_dir, num_usable, status)


def list_nmf_simulations_and_num_frames(
    nmf_rendering_dir: Path, buffer_filenames: dict[str, str], num_workers: int
) -> dict[Path, int]:
    """Index NMF subsegments that have EVERY requested buffer video and a mask.

    Returns a dict mapping each usable subsegment directory to the number of
    frames that are present in all of its buffers simultaneously. Subsegments
    that are missing any buffer are reported as warnings and excluded from the
    pool.
    """
    print("Indexing frames from NeuroMechFly simulation subsegments:")
    subsegment_dirs: list[Path] = []
    for trajectory_dir in nmf_rendering_dir.iterdir():
        if not trajectory_dir.is_dir():
            continue
        for segment_dir in trajectory_dir.iterdir():
            if not segment_dir.is_dir() or not segment_dir.name.startswith("segment_"):
                continue
            for subsegment_dir in segment_dir.iterdir():
                if not subsegment_dir.is_dir() or not subsegment_dir.name.startswith(
                    "subsegment_"
                ):
                    continue
                subsegment_dirs.append(subsegment_dir)

    num_frames_by_subsegment: dict[Path, int] = {}
    skipped: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_subsegment = {
            executor.submit(_count_usable_nmf_frames, subsegment_dir, buffer_filenames): subsegment_dir
            for subsegment_dir in subsegment_dirs
        }
        for future in tqdm(
            as_completed(future_to_subsegment), total=len(future_to_subsegment), disable=None
        ):
            subsegment_dir, num_usable, status = future.result()
            if num_usable <= 0:
                skipped.append((subsegment_dir, status))
                continue
            if status != "ok":
                logging.warning(f"{subsegment_dir}: {status}")
            num_frames_by_subsegment[subsegment_dir] = num_usable

    for subsegment_dir, num_frames in num_frames_by_subsegment.items():
        print(f"  {subsegment_dir}: {num_frames} usable frames")
    for subsegment_dir, status in skipped:
        logging.warning(f"Skipping {subsegment_dir}: {status}")
    if skipped:
        print(
            f"Excluded {len(skipped)} subsegment(s) from the NMF pool due to "
            "missing or invalid buffer/mask data."
        )
    return num_frames_by_subsegment


def _read_foreground_masks(
    subsegment_dir: Path, frame_indices: list[int]
) -> dict[int, np.ndarray]:
    """Load binary silhouette masks for the requested frames from the h5.

    ``postprocessed/segmentation_labels`` is integer-labelled with 0=background,
    so we binarize to a uint8 silhouette aligned with the rendered frames. On any
    failure we return an empty dict and the caller skips every frame of this
    subsegment (preserving the all-or-nothing invariant).
    """
    h5_path = subsegment_dir / "processed_simulation_data.h5"
    if not h5_path.is_file():
        logging.warning(
            f"Expected segmentation h5 {h5_path} does not exist; skipping all "
            "frames from this subsegment to preserve alignment."
        )
        return {}
    unique_sorted = sorted(set(frame_indices))
    masks_by_frame: dict[int, np.ndarray] = {}
    try:
        with h5py.File(h5_path, "r") as h5_file:
            segmentation_dataset = h5_file["postprocessed/segmentation_labels"]
            segmentations = np.asarray(segmentation_dataset[unique_sorted])
        for frame_index, segmentation in zip(unique_sorted, segmentations):
            masks_by_frame[frame_index] = (segmentation != 0).astype(np.uint8) * 255
    except (OSError, KeyError, IndexError, ValueError) as exc:
        logging.warning(
            f"Failed to read masks from {h5_path} ({exc}); skipping all frames "
            "from this subsegment to preserve alignment."
        )
        return {}
    return masks_by_frame


def _extract_one_nmf_subsegment(
    subsegment_dir: Path,
    specs: list[tuple[int, str]],
    buffer_filenames: dict[str, str],
    output_dir: Path,
) -> tuple[int, int]:
    """Extract every selected frame of every buffer for a single subsegment.

    Runs in a worker process. ``specs`` is a list of ``(frame_index,
    split_prefix)``. Returns ``(num_frames_written, num_frames_skipped)``, where
    a "frame written" means all buffers plus the mask were written with a shared
    stem.
    """
    frame_indices = [frame_index for frame_index, _ in specs]

    # Decode every buffer video once at the requested frame indices.
    frames_by_buffer: dict[str, dict[int, np.ndarray]] = {}
    for buffer_key, filename in buffer_filenames.items():
        video_path = subsegment_dir / filename
        try:
            frames, _fps = read_frames_from_video(video_path, frame_indices)
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                f"Failed to decode buffer '{buffer_key}' from {video_path} "
                f"({exc}); skipping all frames from this subsegment."
            )
            return 0, len(specs)
        frames_by_buffer[buffer_key] = {
            frame_index: frame for frame_index, frame in zip(frame_indices, frames)
        }

    masks_by_frame = _read_foreground_masks(subsegment_dir, frame_indices)

    trial, segment_id, subsegment_id = subsegment_dir.parts[-3:]
    num_written = 0
    num_skipped = 0
    for frame_index, split_prefix in specs:
        # All-or-nothing: require the mask and every buffer for this frame.
        if frame_index not in masks_by_frame or any(
            frame_index not in frames_by_buffer[buffer_key]
            for buffer_key in buffer_filenames
        ):
            logging.warning(
                f"Skipping {subsegment_dir} frame {frame_index}: missing mask or buffer."
            )
            num_skipped += 1
            continue

        stem = f"{trial}_{segment_id}_{subsegment_id}_frame_{frame_index:06d}"

        # Primary image as JPEG (matches the previous trainA convention); every
        # prior as lossless PNG so no extra compression is stacked on top of the
        # h264 the buffers were rendered with.
        for buffer_key in buffer_filenames:
            buffer_output_dir = _domain_a_output_dir(output_dir, split_prefix, buffer_key)
            buffer_output_dir.mkdir(parents=True, exist_ok=True)
            extension = "jpg" if buffer_key == PRIMARY_BUFFER_KEY else "png"
            imageio.imwrite(
                buffer_output_dir / f"{stem}.{extension}",
                frames_by_buffer[buffer_key][frame_index],
            )

        mask_output_dir = _domain_a_output_dir(output_dir, split_prefix, "mask")
        mask_output_dir.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(mask_output_dir / f"{stem}.png", masks_by_frame[frame_index])
        num_written += 1

    return num_written, num_skipped


def extract_nmf_simulation_frames_from_specs(
    frame_specs: list[tuple[Path, int, str]],
    buffer_filenames: dict[str, str],
    output_dir: Path,
    num_workers: int,
) -> None:
    """Extract all buffers for the selected NMF frames.

    Args:
        frame_specs: list of ``(subsegment_dir, frame_index, split_prefix)``.
        buffer_filenames: buffer key -> rendered video filename to extract.
        output_dir: dataset root under which the CUT-style folders are created.
        num_workers: number of worker processes (decode + encode is CPU bound).
    """
    specs_by_subsegment: dict[Path, list[tuple[int, str]]] = defaultdict(list)
    for subsegment_dir, frame_index, split_prefix in frame_specs:
        specs_by_subsegment[subsegment_dir].append((frame_index, split_prefix))
    for subsegment_dir in specs_by_subsegment:
        specs_by_subsegment[subsegment_dir].sort(key=lambda item: item[0])

    print("Extracting multi-prior frames from NeuroMechFly subsegments...")
    items = list(specs_by_subsegment.items())
    total_written = 0
    total_skipped = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _extract_one_nmf_subsegment, subsegment_dir, specs, buffer_filenames, output_dir
            )
            for subsegment_dir, specs in items
        ]
        for future in tqdm(as_completed(futures), total=len(futures), disable=None):
            num_written, num_skipped = future.result()
            total_written += num_written
            total_skipped += num_skipped

    buffer_folder_names = ", ".join(
        _domain_a_output_dir(Path("."), "train", key).name
        for key in list(buffer_filenames) + ["mask"]
    )
    print(
        f"NMF extraction: wrote {total_written} aligned frame set(s) across "
        f"[{buffer_folder_names}]"
        + (f", skipped {total_skipped} due to missing buffers/masks" if total_skipped else "")
        + "."
    )
    if total_skipped:
        logging.warning(
            f"{total_skipped} NMF frame(s) were skipped at extraction time even "
            "though the upstream listing marked them usable. The output dataset is "
            "short by that many frames; re-run with a different --random_seed to "
            "draw replacements from the same pool, or investigate the warnings above."
        )


@dataclass
class Config:
    """Command-line configuration for multi-prior dataset extraction."""

    # Directory of NeuroMechFly simulation renderings (the simulated data),
    # generated by poseforge/neuromechfly/scripts/run_simulation.py.
    nmf_rendering_dir: Path = Path("/scratch/stimpfli/poseforge/data/bulk_data")
    # Directory of recorded Spotlight images (domain B, no priors).
    spotlight_recordings_dir: Path = Path(
        "/scratch/stimpfli/poseforge/behavior_images/spotlight_aligned_and_cropped"
    )
    # Output directory for the extracted training dataset.
    output_dir: Path = Path(
        "/scratch/stimpfli/poseforge/clean_datasets/datasets/"
        "aymanns2022_multiprior_spotlight_dataset"
    )
    # Buffer key -> rendered video filename to extract per subsegment. The
    # "image" key is the primary domain-A image; all others are geometric priors.
    simulation_buffer_videos: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SIMULATION_BUFFER_VIDEOS)
    )
    # Number of domain-A (NMF) frames to draw for each split.
    num_train_frames: int = 20000
    num_val_frames: int = 100
    # Number of parallel workers (processes for decode/encode, threads for I/O).
    num_workers: int = os.cpu_count() or 1
    # Random seed for reproducible frame selection.
    random_seed: int = 42


def main(config: Config) -> None:
    buffer_filenames = config.simulation_buffer_videos
    if PRIMARY_BUFFER_KEY not in buffer_filenames:
        raise ValueError(
            f"simulation_buffer_videos must contain the primary '{PRIMARY_BUFFER_KEY}' "
            f"key; got keys {list(buffer_filenames)}."
        )
    output_dir = config.output_dir

    np.random.seed(config.random_seed)

    num_frames_config = {
        "train": {"nmf_simulation": config.num_train_frames, "spotlight_recording": config.num_train_frames},
        "val": {"nmf_simulation": config.num_val_frames, "spotlight_recording": config.num_val_frames},
    }

    # Index available frames.
    nmf_num_frames = list_nmf_simulations_and_num_frames(
        config.nmf_rendering_dir, buffer_filenames, config.num_workers
    )
    spotlight_num_frames = list_spotlight_recordings_and_num_frames(
        config.spotlight_recordings_dir
    )

    # Build flat (source, frame_index) pools and shuffle.
    nmf_frame_pool: list[tuple[Path, int]] = []
    for subsegment_dir, num_frames in nmf_num_frames.items():
        nmf_frame_pool.extend([(subsegment_dir, i) for i in range(num_frames)])
    spotlight_frame_pool: list[tuple[Path, int]] = []
    for trial_dir, num_frames in spotlight_num_frames.items():
        spotlight_frame_pool.extend([(trial_dir, i) for i in range(num_frames)])

    total_nmf_requested = (
        num_frames_config["train"]["nmf_simulation"]
        + num_frames_config["val"]["nmf_simulation"]
    )
    total_spotlight_requested = (
        num_frames_config["train"]["spotlight_recording"]
        + num_frames_config["val"]["spotlight_recording"]
    )
    print(
        f"NMF pool: {len(nmf_frame_pool)} frames available, {total_nmf_requested} requested.\n"
        f"Spotlight pool: {len(spotlight_frame_pool)} frames available, "
        f"{total_spotlight_requested} requested."
    )
    if len(nmf_frame_pool) < total_nmf_requested:
        raise ValueError(
            f"Not enough NeuroMechFly simulation frames available: requested "
            f"{total_nmf_requested}, but only {len(nmf_frame_pool)} frames are available."
        )
    if len(spotlight_frame_pool) < total_spotlight_requested:
        raise ValueError(
            f"Not enough Spotlight recording frames available: requested "
            f"{total_spotlight_requested}, but only {len(spotlight_frame_pool)} available."
        )

    np.random.shuffle(nmf_frame_pool)
    np.random.shuffle(spotlight_frame_pool)

    selected_frame_specs = {
        "train": {"nmf_simulation": [], "spotlight_recording": []},
        "val": {"nmf_simulation": [], "spotlight_recording": []},
    }
    for dataset_type, pool in [
        ("nmf_simulation", nmf_frame_pool),
        ("spotlight_recording", spotlight_frame_pool),
    ]:
        num_train = num_frames_config["train"][dataset_type]
        num_val = num_frames_config["val"][dataset_type]
        selected_frame_specs["train"][dataset_type] = pool[:num_train]
        selected_frame_specs["val"][dataset_type] = pool[num_train : num_train + num_val]

    # CUT uses "train"/"test" folder prefixes; our val split maps to "test".
    dataset_type_to_cut_prefix = {"train": "train", "val": "test"}

    # Domain A (NMF simulation) with all priors.
    nmf_specs: list[tuple[Path, int, str]] = []
    for dataset_type in ["train", "val"]:
        split_prefix = dataset_type_to_cut_prefix[dataset_type]
        for subsegment_dir, frame_index in selected_frame_specs[dataset_type]["nmf_simulation"]:
            nmf_specs.append((subsegment_dir, frame_index, split_prefix))
    extract_nmf_simulation_frames_from_specs(
        nmf_specs, buffer_filenames, output_dir, config.num_workers
    )

    # Domain B (real Spotlight recordings) -- images only, no priors.
    spotlight_specs: list[tuple[Path, int, Path]] = []
    for dataset_type in ["train", "val"]:
        split_prefix = dataset_type_to_cut_prefix[dataset_type]
        image_output_dir = output_dir / f"{split_prefix}B"
        for trial_dir, frame_index in selected_frame_specs[dataset_type]["spotlight_recording"]:
            spotlight_specs.append((trial_dir, frame_index, image_output_dir))
    extract_spotlight_recording_frames_from_specs(spotlight_specs, config.num_workers)


if __name__ == "__main__":
    main(tyro.cli(Config))
