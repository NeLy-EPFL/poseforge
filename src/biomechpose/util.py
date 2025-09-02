import random
import torch
import numpy as np
import os
import psutil
import warnings
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

    print(f"Random seed set to {seed} for reproducible results")


def configure_matplotlib_style():
    import matplotlib
    import logging

    matplotlib.style.use("fast")
    plt.rcParams["font.family"] = "Arial"
    # suppress matplotlib font manager warnings
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


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
