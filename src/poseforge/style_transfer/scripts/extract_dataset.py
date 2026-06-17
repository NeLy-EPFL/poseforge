import os
import h5py
import numpy as np
import pandas as pd
import logging
import imageio.v2 as imageio
import tyro
from shutil import copyfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path
from pvio.io import read_frames_from_video, check_num_frames


def list_nmf_simulations_and_num_frames(
    nmf_rendering_dir: Path, video_filename: str, num_workers: int
) -> dict[Path, int]:
    print("Indexing frames from NeuroMechFly simulation videos:")
    # Collect all candidate video files first, then count frames in parallel
    # (each check_num_frames opens the video, which is I/O bound).
    video_files = []
    for traj_dir in nmf_rendering_dir.iterdir():
        if not traj_dir.is_dir():
            continue
        for seg_dir in traj_dir.iterdir():
            if not seg_dir.is_dir() or not seg_dir.name.startswith("segment_"):
                continue
            for subseg_dir in seg_dir.iterdir():
                if not subseg_dir.is_dir() or not subseg_dir.name.startswith(
                    "subsegment_"
                ):
                    continue
                video_file = subseg_dir / video_filename
                if not video_file.is_file():
                    logging.warning(f"Expected video file {video_file} does not exist.")
                video_files.append(video_file)

    num_frames_dict = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_video = {
            executor.submit(check_num_frames, vf): vf for vf in video_files
        }
        for future in tqdm(
            as_completed(future_to_video), total=len(future_to_video), disable=None
        ):
            video_file = future_to_video[future]
            num_frames_dict[video_file] = future.result()
    for video_file, num_frames in num_frames_dict.items():
        print(f"  {video_file}: {num_frames} frames")
    return num_frames_dict


def list_spotlight_recordings_and_num_frames(
    spotlight_recordings_dir: Path,
) -> dict[Path, int]:
    print("Indexing frames from Spotlight recordings:")
    num_frames_dict = {}
    for trial_dir in spotlight_recordings_dir.iterdir():
        if not trial_dir.is_dir():
            continue
        flip_label_file = trial_dir / "predicted_flip_labels.csv"
        df = pd.read_csv(flip_label_file)
        df = df[df["predicted_label"] == "not flipped"]
        num_frames_dict[trial_dir] = len(df)
        print(f"  {trial_dir}: {num_frames_dict[trial_dir]} frames")
    return num_frames_dict


def _extract_one_nmf_video(
    video_path: Path, specs: list[tuple[int, Path]]
) -> int:
    """Extract every selected frame (and mask) for a single NMF video.

    Runs in a worker process; everything it touches is independent of the
    other videos. Returns the number of frames written.
    """
    frame_indices = [frame_idx for frame_idx, _ in specs]
    frames, fps = read_frames_from_video(video_path, frame_indices)
    frames_dict = {idx: frame for idx, frame in zip(frame_indices, frames)}

    # Load the matching foreground masks from the simulation h5 once per
    # video. "postprocessed/segmentation_labels" is integer-labelled with
    # 0=background, so we binarize to a uint8 silhouette mask aligned with
    # the rendered video frames. Read all needed frames in a single sorted,
    # de-duplicated fancy-index call instead of one slice per frame.
    h5_path = video_path.parent / "processed_simulation_data.h5"
    masks_dict = {}
    if h5_path.is_file():
        unique_sorted = sorted(set(frame_indices))
        with h5py.File(h5_path, "r") as h5_file:
            seg_dataset = h5_file["postprocessed/segmentation_labels"]
            segs = np.asarray(seg_dataset[unique_sorted])
        for frame_idx, seg in zip(unique_sorted, segs):
            masks_dict[frame_idx] = (seg != 0).astype(np.uint8) * 255
    else:
        logging.warning(
            f"Expected segmentation h5 file {h5_path} does not exist; "
            "masks will not be extracted for this video."
        )

    trial, segment_id, subsegment_id = str(video_path.parent).split("/")[-3:]
    for frame_idx, output_dir in specs:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{trial}_{segment_id}_{subsegment_id}_frame_{frame_idx:06d}"
        output_path = output_dir / f"{stem}.jpg"
        imageio.imwrite(output_path, frames_dict[frame_idx])

        if frame_idx in masks_dict:
            # Mirror the trainA/ vs testA/ layout with trainA_mask/ vs testA_mask/.
            mask_dir = output_dir.parent / f"{output_dir.name}_mask"
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = mask_dir / f"{stem}.png"
            imageio.imwrite(mask_path, masks_dict[frame_idx])
    return len(specs)


def extract_nmf_simulation_frames_from_specs(
    frame_specs: list[tuple[Path, int, Path]], num_workers: int
):
    """Extract frames from NeuroMechFly simulation videos based on spec
    list (defined below).

    Args:
        frame_specs (list[tuple[Path, int, Path]]): A list of tuples, each
            containing (i) the video path, (ii) frame index, and (iii)
            directory under which the frame should be stored.
        num_workers (int): Number of worker processes to fan videos out to.
    """
    # Sort selected frames by video and frame idx within video
    specs_by_video = defaultdict(list)  # video_path -> list of (frame_idx, output_dir)
    for video_path, frame_idx, output_dir in frame_specs:
        specs_by_video[video_path].append((frame_idx, output_dir))
    for key, val in specs_by_video.items():
        specs_by_video[key] = sorted(val, key=lambda x: x[0])

    # Extract frames. Each video is fully independent, and the work (video
    # decode + JPEG/PNG encode) is CPU bound, so fan the videos out across
    # worker processes.
    print("Extracting frames from NeuroMechFly simulation videos...")
    items = list(specs_by_video.items())
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_extract_one_nmf_video, video_path, specs)
            for video_path, specs in items
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), disable=None
        ):
            future.result()


def extract_spotlight_recording_frames_from_specs(
    frame_specs: list[tuple[Path, int, Path]], num_workers: int
):
    """Extract frames from Spotlight recordings based on spec list (defined
    below).

    Args:
        frame_specs (list[tuple[Path, int, Path]]): A list of tuples, each
            containing (i) the video path, (ii) frame index, and (iii)
            directory under which the frame should be stored.
        num_workers (int): Number of I/O worker threads for the file copies.
    """
    # Sort selected frames by video and frame idx within video
    specs_by_subsegment = defaultdict(
        list
    )  # video_path -> list of (frame_idx, output_dir)
    for video_path, frame_idx, output_dir in frame_specs:
        specs_by_subsegment[video_path].append((frame_idx, output_dir))
    for key, val in specs_by_subsegment.items():
        specs_by_subsegment[key] = sorted(val, key=lambda x: x[0])

    # Build the full list of (src, dst) copy jobs, then run them in a thread
    # pool (pure file I/O, so threads are enough and avoid pickling overhead).
    print("Extracting frames from Spotlight recordings...")
    copy_jobs = []  # list of (input_path, output_path)
    for trial_dir, specs in specs_by_subsegment.items():
        dataframe = pd.read_csv(trial_dir / "predicted_flip_labels.csv")
        dataframe = dataframe[dataframe["predicted_label"] == "not flipped"]
        for frame_idx_among_selection, out_dir in specs:
            row = dataframe.iloc[frame_idx_among_selection]
            image_filename = row["image"]
            input_path = trial_dir / "all" / image_filename
            output_path = out_dir / f"{trial_dir.name}_{image_filename}"
            out_dir.mkdir(parents=True, exist_ok=True)
            copy_jobs.append((input_path, output_path))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(copyfile, src, dst) for src, dst in copy_jobs
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), disable=None
        ):
            future.result()


@dataclass
class Config:
    """Command-line configuration for dataset extraction."""

    # Directory of NeuroMechFly simulation renderings (the simulated data),
    # generated by poseforge/neuromechfly/scripts/run_simulation.py.
    nmf_rendering_dir: Path = Path("/scratch/stimpfli/poseforge/data/bulk_data")
    # Directory of recorded Spotlight images.
    spotlight_recordings_dir: Path = Path(
        "/scratch/stimpfli/poseforge/behavior_images/spotlight_aligned_and_cropped"
    )
    # Output directory for the extracted training images.
    output_dir: Path = Path(
        "/scratch/stimpfli/poseforge/clean_datasets/datasets/aymanns2022_pseudocolor_spotlight_dataset_gray"
    )
    # Name of the rendered video file inside each NMF subsegment directory.
    video_filename: str = "processed_nmf_sim_render_grayscale.mp4"
    # Number of parallel workers (processes for decode/encode, threads for I/O).
    num_workers: int = os.cpu_count() or 1
    # Random seed for reproducible frame selection.
    random_seed: int = 42


def main(config: Config):
    nmf_rendering_dir = config.nmf_rendering_dir
    spotlight_recordings_dir = config.spotlight_recordings_dir
    output_dir = config.output_dir
    video_filename = config.video_filename
    num_workers = config.num_workers

    # Fix random state for reproducibility
    np.random.seed(config.random_seed)

    # Define number of frames to extract
    num_frames_config = {
        "train": {"nmf_simulation": 10000, "spotlight_recording": 10000},
        "val": {"nmf_simulation": 1000, "spotlight_recording": 1000},
    }

    # Index number of frames in each trial/recording
    # NMF simulation
    nmf_num_frames = list_nmf_simulations_and_num_frames(
        nmf_rendering_dir, video_filename, num_workers
    )
    print("NMF Simulation Video Frames:")
    for video, num_frames in nmf_num_frames.items():
        print(f"  {video}: {num_frames} frames")

    # Spotlight recordings
    spotlight_num_frames = list_spotlight_recordings_and_num_frames(
        spotlight_recordings_dir
    )

    # Randomly select frames
    nmf_frame_configs = []
    for trial, num_frames in nmf_num_frames.items():
        nmf_frame_configs.extend([(trial, i) for i in range(num_frames)])
    spotlight_frame_configs = []
    for trial, num_frames in spotlight_num_frames.items():
        spotlight_frame_configs.extend([(trial, i) for i in range(num_frames)])
    total_num_nmf_frames_requested = (
        num_frames_config["train"]["nmf_simulation"]
        + num_frames_config["val"]["nmf_simulation"]
    )
    total_num_spotlight_frames_requested = (
        num_frames_config["train"]["spotlight_recording"]
        + num_frames_config["val"]["spotlight_recording"]
    )
    if len(nmf_frame_configs) < total_num_nmf_frames_requested:
        raise ValueError(
            f"Not enough NeuroMechFly simulation frames available: "
            f"requested {num_frames_config['train']['nmf_simulation']} training frames "
            f"and {num_frames_config['val']['nmf_simulation']} validation frames, "
            f"but only {len(nmf_frame_configs)} frames are available."
        )
    if len(spotlight_frame_configs) < total_num_spotlight_frames_requested:
        raise ValueError(
            f"Not enough Spotlight recording frames available: "
            f"requested {num_frames_config['train']['spotlight_recording']} training "
            f"frames and {num_frames_config['val']['spotlight_recording']} validation, "
            f"frames, but only {len(spotlight_frame_configs)} frames are available."
        )
    np.random.shuffle(nmf_frame_configs)
    np.random.shuffle(spotlight_frame_configs)
    selected_frame_specs = {
        "train": {"nmf_simulation": [], "spotlight_recording": []},
        "val": {"nmf_simulation": [], "spotlight_recording": []},
    }
    for dataset_type, specs in [
        ("nmf_simulation", nmf_frame_configs),
        ("spotlight_recording", spotlight_frame_configs),
    ]:
        num_train = num_frames_config["train"][dataset_type]
        num_val = num_frames_config["val"][dataset_type]
        train_selection = specs[:num_train]
        val_selection = specs[num_train : num_train + num_val]
        selected_frame_specs["train"][dataset_type] = train_selection
        selected_frame_specs["val"][dataset_type] = val_selection

    # Create file structure as expected by CUT
    dataset_type_to_cut_dataset_type = {"train": "train", "val": "test"}
    # NeuroMechFly simulation
    nmf_specs = []
    for dataset_type in ["train", "val"]:
        dataset_type_ = dataset_type_to_cut_dataset_type[dataset_type]
        for path, frame_idx in selected_frame_specs[dataset_type]["nmf_simulation"]:
            img_out_dir = output_dir / f"{dataset_type_}A"
            nmf_specs.append((path, frame_idx, img_out_dir))
    extract_nmf_simulation_frames_from_specs(nmf_specs, num_workers)

    # Spotlight recordings
    spotlight_specs = []
    for dataset_type in ["train", "val"]:
        dataset_type_ = dataset_type_to_cut_dataset_type[dataset_type]
        for path, frame_idx in selected_frame_specs[dataset_type][
            "spotlight_recording"
        ]:
            img_out_dir = output_dir / f"{dataset_type_}B"
            spotlight_specs.append((path, frame_idx, img_out_dir))
    extract_spotlight_recording_frames_from_specs(spotlight_specs, num_workers)


if __name__ == "__main__":
    main(tyro.cli(Config))
