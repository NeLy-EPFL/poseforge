"""TODO: Eventually get rid of this and use pvio directly."""

import numpy as np
import pvio.video_io as video_io
from pathlib import Path


def read_frames_from_video(
    video_path: Path, frame_indices: list[int] | None = None
) -> tuple[list[np.ndarray], float]:
    return video_io.read_frames_from_video(video_path, frame_indices)


default_video_writing_ffmpeg_params = video_io._default_ffmpeg_params_for_video_writing


def check_num_frames(video_path: Path) -> int:
    return video_io.check_num_frames(video_path)


def round_up_to_multiple(x: int, multiple_of: int) -> int:
    return ((x + multiple_of - 1) // multiple_of) * multiple_of
