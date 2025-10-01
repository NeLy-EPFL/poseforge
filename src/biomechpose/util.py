import random
import torch
import numpy as np
import pandas as pd
import os
import psutil
import gc
import logging
import imageio.v2 as imageio
from pathlib import Path
from typing import Any, Iterator
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

    # Make torch deterministic
    # The following env var must be set for CUDA >= 10.2). See
    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

    logging.info(f"Random seed set to {seed} for reproducible results")


def configure_matplotlib_style():
    import matplotlib
    import logging

    matplotlib.style.use("fast")
    plt.rcParams["font.family"] = "Arial"
    # suppress matplotlib font manager warnings
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def get_hardware_availability(
    check_gpu: bool = False, print_results: bool = False
) -> dict[str, Any]:
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


def clear_memory_cache(logging_level=logging.DEBUG):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Log current GPU memory usage
        allocated = torch.cuda.memory_allocated() / 1000**3  # GB
        cached = torch.cuda.memory_reserved() / 1000**3  # GB
        logging.log(
            logging_level,
            f"GPU memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached",
        )


def check_num_frames(video_path: Path) -> int:
    """Check number of frames in a video file using imageio.v2"""
    try:
        with imageio.get_reader(video_path) as reader:
            num_frames = reader.count_frames()
    except Exception as e:
        raise RuntimeError(f"Failed to open video file: {video_path}") from e
    return num_frames


def round_up_to_multiple(x: int, multiple_of: int) -> int:
    return ((x + multiple_of - 1) // multiple_of) * multiple_of


def check_mixed_precision_status(
    use_float16: bool,
    device: torch.device,
    tensors: dict[str, torch.Tensor | Iterator[torch.Tensor]] | None = None,
    amp_scaler: torch.amp.GradScaler | None = None,
    print_results: bool = False,
    subtitle: str | None = None,
) -> dict:
    """Check if the pipeline is configured for mixed precision and get
    current status on given tensors.

    Args:
        use_float16 (bool): Whether mixed precision is intended to be used.
        device (torch.device): Device where tensors are located.F
        tensors (dict[str, torch.Tensor | Iterator[torch.Tensor]] | None):
            Optional dictionary of tensors or iterables of tensors to check dtypes
            for. Note that these can also be model parameters.
        amp_scaler (torch.amp.GradScaler | None): Optional GradScaler to
            check status for (enabled vs disabled).
        print_results (bool): If True, print results to console.
        subtitle (str | None): Optional subtitle to print in the header.

    Returns:
        dict: Status information about mixed precision status.
    """
    status = {
        "use_float16_flag": use_float16,
        "device": str(device),
        "device_supports_half": torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 7,
    }
    if amp_scaler is not None:
        status["amp_scaler_enabled"] = amp_scaler.is_enabled()

    # Check model parameter dtypes
    if tensors is not None:
        for name, thing in tensors.items():
            if isinstance(thing, torch.Tensor):
                thing = [thing]
            dtypes = set(p.dtype for p in thing)
            status[f"{name}_param_dtypes"] = [str(dt) for dt in sorted(dtypes)]

    # Print out results if requested
    if print_results:
        print("==================== Mixed Precision Status ====================")
        if subtitle is not None:
            print(f"({subtitle})")
        for key, value in status.items():
            print(f"{key}: {value}")
        print("================================================================")

    return status
