import random
import torch
import numpy as np
import os
import gc
import logging
import sys
from loguru import logger
from psutil import Process
from sys import stderr
from typing import Iterator, Any


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

    logger.info(f"Random seed set to {seed} for reproducible results")


def get_hardware_availability(
    check_gpu: bool = False, print_results: bool = False
) -> dict[str, Any]:
    """Print available CPU and GPU cores"""
    res = {}

    num_cpu_cores_available = len(Process().cpu_affinity())
    num_cpu_cores_total = os.cpu_count()
    res["num_cpu_cores_available"] = num_cpu_cores_available
    res["num_cpu_cores_total"] = num_cpu_cores_total
    if print_results:
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
            if print_results:
                print(f"CUDA is available. GPUs:")
                for i, name in enumerate(gpu_names):
                    print(f"  GPU {i}: {name}")
            res["gpus"] = gpu_names
        else:
            if print_results:
                print("CUDA is not available.")
            res["gpus"] = []
    else:
        res["gpus"] = None

    return res


def clear_memory_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Log current GPU memory usage
        allocated = torch.cuda.memory_allocated() / 1000**3  # GB
        cached = torch.cuda.memory_reserved() / 1000**3  # GB
        logger.info(f"GPU memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")


def check_mixed_precision_status(
    use_float16: bool,
    device: torch.device,
    tensors: dict[str, torch.Tensor | Iterator[torch.Tensor]] | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
    print_results: bool = False,
    subtitle: str | None = None,
) -> dict:
    """Check if the pipeline is configured for mixed precision and get
    current status on given tensors.

    Args:
        use_float16 (bool): Whether mixed precision is intended to be used.
        device (torch.device): Device where tensors are located.
        tensors (dict[str, torch.Tensor | Iterator[torch.Tensor]] | None):
            Optional dictionary of tensors or iterables of tensors to check dtypes
            for. Note that these can also be model parameters.
        grad_scaler (torch.amp.GradScaler | None): Optional GradScaler to
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
    if grad_scaler is not None:
        status["amp_grad_scaler_enabled"] = grad_scaler.is_enabled()

    # Check model parameter dtypes
    if tensors is not None:
        for name, thing in tensors.items():
            if isinstance(thing, (torch.Tensor, np.ndarray, float, int)):
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


def set_loguru_level(level: str) -> None:
    """Set the logging level for loguru logger and intercept standard logging."""
    # Remove existing loguru handlers
    logger.remove()

    # Add loguru handler with your desired level
    logger.add(sink=sys.stderr, level=level)

    # Silence info or less from matplotlib font manager and fontTools subset (from the normal logging module)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    # Intercept standard logging and redirect to loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=level.upper(), force=True)


class InterceptHandler(logging.Handler):
    """
    Handler that intercepts standard logging calls and redirects them to loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )
