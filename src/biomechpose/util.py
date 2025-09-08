import random
import torch
import numpy as np
import os
import psutil
import imageio.v2 as imageio
from pathlib import Path
from matplotlib import pyplot as plt


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible results"""
    # Python random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for additional determinism
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Generator for DataLoader workers
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"🎲 Random seed set to {seed} for reproducible results")


def configure_matplotlib_style():
    import matplotlib

    matplotlib.style.use("fast")
    plt.rcParams["font.family"] = "Arial"


def print_hardware_availability(check_gpu: bool = False):
    """Print available CPU and GPU cores"""
    res = {}

    num_cpu_cores_available = len(psutil.Process().cpu_affinity())
    num_cpu_cores_total = os.cpu_count()
    res["num_cpu_cores_available"] = num_cpu_cores_available
    res["num_cpu_cores_total"] = num_cpu_cores_total
    print(
        f"CPU cores: {num_cpu_cores_available} available "
        f"out of {num_cpu_cores_total} total"
    )

    if check_gpu:
        is_cuda_available = torch.cuda.is_available()
        if is_cuda_available:
            gpu_names = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            print(f"CUDA is available. GPUs:")
            for i, name in enumerate(gpu_names):
                print(f"  GPU {i}: {name}")
            res["gpus"] = gpu_names
        else:
            print("CUDA is not available.")
            res["gpus"] = []
    else:
        res["gpus"] = None

    return res


def read_frames_from_video(
    video_path: Path, frame_indices: list[int] | None = None
) -> tuple[list[np.ndarray], float]:
    """Read specific frames from a video file.

    Args:
        video_path (Path): Path to the video file.
        frame_indices (list[int] | None): List of frame indices to read.
            If None, read all frames.

    Raises:
        ValueError: If the video file cannot be read.
        IndexError: If the frame indices are invalid.

    Returns:
        frames (list[np.ndarray]): List of frames as numpy arrays.
        fps (float): FPS of the video.
    """
    frames = []
    with imageio.get_reader(video_path) as reader:
        if frame_indices is None:
            frame_indices = list(range(reader.count_frames()))
        for idx in frame_indices:
            frames.append(reader.get_data(idx))
        fps = reader.get_meta_data().get("fps", None)
    return frames, fps


default_video_writing_ffmpeg_params = [
    "-crf",
    "15",  # Lower CRF = higher quality (15 is very high quality)
    "-preset",
    "slow",  # Slower preset = better compression efficiency
    "-profile:v",
    "high",  # Use high profile for better compression
    "-level",
    "4.0",  # H.264 level
]
