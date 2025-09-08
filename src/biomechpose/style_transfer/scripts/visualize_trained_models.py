import logging
import json
import cv2
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from biomechpose.util import read_frames_from_video, default_video_writing_ffmpeg_params


def index_visualized_videos(
    run_dir: Path,
) -> tuple[dict[int, dict[int, Path]], dict[str, Path], dict[str, any]]:
    """Index visualized videos by example simulation and epoch number.

    Returns:
        A nested dictionary where the outer key is the name of the example
            simulation, the inner key is the epoch number, and the value
            is the Path to the video file.
        A dictionary mapping simulation names to their full paths.
        A dictionary of hyperparameters from the metadata file.
    """
    # Read metadata
    with open(run_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    hparams = metadata["hyperparameters"]
    hparams["run_name"] = metadata.get("run_name", "unknown_run")

    # Scan visualized videos
    visualized_videos_paths = defaultdict(dict)
    example_sim_path_lookup = {}
    for path in run_dir.glob("*.mp4"):
        parts = path.stem.split("_")
        if len(parts) != 2:
            logging.warning(f"Unexpected file name format: {path}. Skipping.")
            continue
        epoch_str = parts[0].replace("epoch", "")
        example_sim_str = parts[1].replace("examplesim", "")
        try:
            epoch = int(epoch_str)
            example_sim_idx = int(example_sim_str)
        except ValueError:
            logging.warning(f"Unexpected file name format: {path}. Skipping.")
            continue
        simulation_path = Path(metadata["simulation_data_dirs"][example_sim_idx])
        simulation_name = "/".join(simulation_path.parts[-3:])
        visualized_videos_paths[simulation_name][epoch] = path
        example_sim_path_lookup[simulation_name] = simulation_path
    return visualized_videos_paths, example_sim_path_lookup, hparams


def get_cv2_text_scale(target_total_h_px, text, font, thickness):
    (_, h), base = cv2.getTextSize(text, font, 1.0, thickness)
    return target_total_h_px / float(h + base)


def draw_frame(
    canvas_size_width_height: tuple[int, int],
    simulated_frame: np.array,
    styled_frames: list[np.array],
    video_top_left_corner_xy: tuple[int, int],
    num_cols_excluding_original: int,
    simulation_name: str,
    hparams: dict[str, any],
    styled_frames_names: list[str],
    text_area_height: int,
):
    # Calculate layout parameters
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 2
    main_text_height = int(0.3 * text_area_height)
    main_text_scale_cv2 = get_cv2_text_scale(
        main_text_height, "X", text_font, text_thickness
    )

    # Adjust thickness for larger text
    if main_text_scale_cv2 > 2.0:
        text_thickness = 3

    # Create blank black image
    # Ensure width and height are multiples of 16 for video compression reasons
    width = ((canvas_size_width_height[0] + 15) // 16) * 16
    height = ((canvas_size_width_height[1] + 15) // 16) * 16
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Add title of simulation
    x = int(main_text_height * 0.5)
    y = int(main_text_height * 1.3)
    cv2.putText(
        image,
        f"Simulation: {simulation_name}",
        (x, y),
        text_font,
        main_text_scale_cv2,
        (255, 0, 0),
        text_thickness,
    )

    # Add hyperparameters
    x = int(main_text_height * 0.5)
    y = int(main_text_height * 2.6)
    hparams_strs = []
    for key, val in hparams.items():
        hparams_strs.append(f"{key}={val}")
    hparams_str = ", ".join(hparams_strs)
    cv2.putText(
        image,
        f"Hyperparameters: {hparams_str}",
        (x, y),
        text_font,
        main_text_scale_cv2,
        (255, 255, 255),
        text_thickness,
    )

    # Paste simulated frame
    x, y = video_top_left_corner_xy
    h, w, _ = simulated_frame.shape
    image[y : y + h, x : x + w] = simulated_frame

    # Draw styled frames
    for i, (styled_frame, name) in enumerate(zip(styled_frames, styled_frames_names)):
        panel_row = i // num_cols_excluding_original
        panel_col = i % num_cols_excluding_original + 1
        x = video_top_left_corner_xy[0] + panel_col * w
        y = video_top_left_corner_xy[1] + panel_row * h
        image[y : y + h, x : x + w] = styled_frame
        cv2.putText(
            image,
            name,
            (int(x + main_text_height * 0.5), int(y + main_text_height * 1.0)),
            text_font,
            main_text_scale_cv2 * 0.6,
            (255, 255, 255),
            text_thickness,
        )

    return image


def generate_summary_video_for_styled_videos(
    simulated_video_path: Path,
    styled_video_paths: dict[int, Path],
    output_path: Path,
    num_cols_excluding_original: int = 5,
    simulation_name: str = "",
    hparams: dict[str, any] = {},
):
    # Read the simulated video frames
    simulated_frames, fps = read_frames_from_video(simulated_video_path)

    # Read the styled video frames
    styled_frames_dict = {}
    for epoch in sorted(styled_video_paths.keys()):
        video_path = styled_video_paths[epoch]
        styled_frames, my_fps = read_frames_from_video(video_path)
        styled_frames_dict[epoch] = styled_frames
        if fps != my_fps:
            logging.warning(
                f"FPS mismatch between simulated video ({fps}) and "
                f"styled video {video_path} ({my_fps}). Using {fps}."
            )
        if len(styled_frames) != len(simulated_frames):
            raise RuntimeError(
                f"Frame count mismatch between simulated video "
                f"({len(simulated_frames)}) and styled video "
                f"{video_path} ({len(styled_frames)}). Using {len(simulated_frames)}."
            )

    # Resize simulated video to have the same shape as the styled videos using high-quality interpolation
    target_shape = styled_frames[0].shape
    simulated_frames_resized = [
        cv2.resize(
            frame, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LANCZOS4
        )
        for frame in simulated_frames
    ]

    # Figure out canvas parameters
    per_video_height, per_video_width, _ = simulated_frames_resized[0].shape
    num_rows = (
        len(styled_video_paths) + num_cols_excluding_original - 1
    ) // num_cols_excluding_original + 1
    text_area_height = int(0.3 * per_video_height)
    canvas_height = per_video_height * num_rows + text_area_height
    canvas_width = per_video_width * (num_cols_excluding_original + 1)

    # Create frames one by one
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with imageio.get_writer(
        str(output_path),
        "ffmpeg",
        fps=fps,
        codec="libx264",
        quality=None,  # Use CRF instead of quality
        ffmpeg_params=default_video_writing_ffmpeg_params,
    ) as video_writer:
        for frame_idx in range(len(simulated_frames_resized)):
            simulated_frame = simulated_frames_resized[frame_idx]
            styled_frames = []
            styled_frames_names = []
            for epoch, styled_frames_full_video in styled_frames_dict.items():
                styled_frames.append(styled_frames_full_video[frame_idx])
                styled_frames_names.append(f"Epoch {epoch}")
            image = draw_frame(
                (canvas_width, canvas_height),
                simulated_frame,
                styled_frames,
                (0, text_area_height),
                num_cols_excluding_original,
                simulation_name,
                hparams,
                styled_frames_names,
                text_area_height,
            )
            video_writer.append_data(image)


def make_summary_video(run_dir: Path, output_dir: Path) -> None:
    # Index what's under the run directory
    visualized_videos_paths, sim_name_path, hparams = index_visualized_videos(run_dir)

    # Process each example simulation
    for example_sim_name, dict_epoch_to_path in visualized_videos_paths.items():
        example_sim_dir = sim_name_path[example_sim_name]
        sim_video_path = example_sim_dir / "processed_nmf_sim_render_colorcode_0.mp4"
        if not sim_video_path.is_file():
            raise RuntimeError(
                f"Expected simulation video {sim_video_path} does not exist. "
                f"Skipping."
            )
        generate_summary_video_for_styled_videos(
            sim_video_path,
            dict_epoch_to_path,
            output_dir / example_sim_name.replace("/", "_") / f"{run_dir.name}.mp4",
            num_cols_excluding_original=5,
            simulation_name=example_sim_name,
            hparams=hparams,
        )


if __name__ == "__main__":
    synthetic_output_basedir = Path("bulk_data/style_transfer/synthetic_output")
    summary_output_dir = synthetic_output_basedir / "summary_videos"
    summary_output_dir.mkdir(exist_ok=True, parents=True)

    # Detect input videos
    runs_to_visualize = [
        path
        for path in synthetic_output_basedir.iterdir()
        if path.is_dir() and path.name != "summary_videos"
    ]
    runs_to_visualize = sorted(runs_to_visualize)

    for run_dir in tqdm(runs_to_visualize, disable=None):
        make_summary_video(run_dir, summary_output_dir)
