import numpy as np
import pandas as pd
import logging
import imageio.v2 as imageio
from shutil import copyfile
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

from poseforge.util import read_frames_from_video


def check_num_frames_in_video(video_path: Path) -> int:
    """
    Check the number of frames in a video file.

    Args:
        video_path (Path): Path to the video file.

    Returns:
        int: Number of frames in the video.
    """
    with imageio.get_reader(video_path) as reader:
        return reader.count_frames()


def list_nmf_simulations_and_num_frames(nmf_rendering_dir: Path) -> dict[Path, int]:
    print("Indexing frames from NeuroMechFly simulation videos:")
    num_frames_dict = {}
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
                video_file = subseg_dir / "processed_nmf_sim_render_colorcode_0.mp4"
                if not video_file.is_file():
                    logging.warning(f"Expected video file {video_file} does not exist.")
                num_frames = check_num_frames_in_video(video_file)
                num_frames_dict[video_file] = num_frames
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


def extract_nmf_simulation_frames_from_specs(frame_specs: list[tuple[Path, int, Path]]):
    """Extract frames from NeuroMechFly simulation videos based on spec
    list (defined below).

    Args:
        frame_specs (list[tuple[Path, int, Path]]): A list of tuples, each
            containing (i) the video path, (ii) frame index, and (iii)
            directory under which the frame should be stored.
    """
    # Sort selected frames by video and frame idx within video
    specs_by_video = defaultdict(list)  # video_path -> list of (frame_idx, output_dir)
    for video_path, frame_idx, output_dir in frame_specs:
        specs_by_video[video_path].append((frame_idx, output_dir))
    for key, val in specs_by_video.items():
        specs_by_video[key] = sorted(val, key=lambda x: x[0])

    # Extract frames
    print("Extracting frames from NeuroMechFly simulation videos...")
    for video_path, specs in tqdm(specs_by_video.items(), disable=None):
        frame_indices = [frame_idx for frame_idx, _ in specs]
        frames, fps = read_frames_from_video(video_path, frame_indices)
        frames_dict = {idx: frame for idx, frame in zip(frame_indices, frames)}
        for frame_idx, output_dir in specs:
            output_dir.mkdir(parents=True, exist_ok=True)
            trial, segment_id, subsegment_id = str(video_path.parent).split("/")[-3:]
            output_path = (
                output_dir
                / f"{trial}_{segment_id}_{subsegment_id}_frame_{frame_idx:06d}.jpg"
            )
            imageio.imwrite(output_path, frames_dict[frame_idx])


def extract_spotlight_recording_frames_from_specs(
    frame_specs: list[tuple[Path, int, Path]],
):
    """Extract frames from Spotlight recordings based on spec list (defined
    below).

    Args:
        frame_specs (list[tuple[Path, int, Path]]): A list of tuples, each
            containing (i) the video path, (ii) frame index, and (iii)
            directory under which the frame should be stored.
    """
    # Sort selected frames by video and frame idx within video
    specs_by_subsegment = defaultdict(
        list
    )  # video_path -> list of (frame_idx, output_dir)
    for video_path, frame_idx, output_dir in frame_specs:
        specs_by_subsegment[video_path].append((frame_idx, output_dir))
    for key, val in specs_by_subsegment.items():
        specs_by_subsegment[key] = sorted(val, key=lambda x: x[0])

    # Extract frames
    print("Extracting frames from Spotlight recordings...")
    for trial_dir, specs in tqdm(specs_by_subsegment.items(), disable=None):
        dataframe = pd.read_csv(trial_dir / "predicted_flip_labels.csv")
        dataframe = dataframe[dataframe["predicted_label"] == "not flipped"]
        for frame_idx_among_selection, out_dir in specs:
            row = dataframe.iloc[frame_idx_among_selection]
            image_filename = row["image"]
            input_path = trial_dir / "all" / image_filename
            output_path = out_dir / f"{trial_dir.name}_{image_filename}"
            out_dir.mkdir(parents=True, exist_ok=True)
            copyfile(input_path, output_path)


if __name__ == "__main__":
    # Define data paths
    # Simulated images (generated by poseforge/neuromechfly/scripts/run_simulation.py)
    nmf_rendering_dir = Path("bulk_data/nmf_rendering")
    # Recorded images (from Spotlight)
    spotlight_recordings_dir = Path(
        "bulk_data/behavior_images/spotlight_aligned_and_cropped"
    )
    # Output directory for extracted training images
    output_dir = Path(
        "bulk_data/style_transfer/aymanns2022_pseudocolor_spotlight_dataset"
    )

    # Fix random state for reproducibility
    random_seed = 42
    np.random.seed(random_seed)

    # Define number of frames to extract
    num_frames_config = {
        "train": {"nmf_simulation": 10000, "spotlight_recording": 10000},
        "val": {"nmf_simulation": 1000, "spotlight_recording": 1000},
    }

    # Index number of frames in each trial/recording
    # NMF simulation
    nmf_num_frames = list_nmf_simulations_and_num_frames(nmf_rendering_dir)
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
    extract_nmf_simulation_frames_from_specs(nmf_specs)

    # Spotlight recordings
    spotlight_specs = []
    for dataset_type in ["train", "val"]:
        dataset_type_ = dataset_type_to_cut_dataset_type[dataset_type]
        for path, frame_idx in selected_frame_specs[dataset_type][
            "spotlight_recording"
        ]:
            img_out_dir = output_dir / f"{dataset_type_}B"
            spotlight_specs.append((path, frame_idx, img_out_dir))
    extract_spotlight_recording_frames_from_specs(spotlight_specs)
