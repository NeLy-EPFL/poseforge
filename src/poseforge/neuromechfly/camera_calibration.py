"""Load a camera-calibration artifact exported by spotlight-tools.

The artifact is produced by
``spotlight_tools.calibration.get_postprocess_cameramatrix_equivalent`` and
contains the MuJoCo-ready ``fovy_deg`` and ``pos_offset_z_mm`` that reproduce
the experimental Spotlight camera's pixel-per-mm at the sample plane.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    fovy_deg: float
    pos_offset_z_mm: float
    render_size: int
    pixel_per_mm_at_sample: float
    physical_distance_mm: float


def load_camera_config(path: str | Path) -> CameraConfig:
    """Read a poseforge_camera.npz and return the MuJoCo-ready knobs.

    Args:
        path: Path to the .npz produced by
            ``get_postprocess_cameramatrix_equivalent.py``.
    """
    path = Path(path)
    with np.load(path) as data:
        cfg = CameraConfig(
            fovy_deg=float(data["fovy_deg"]),
            pos_offset_z_mm=float(data["pos_offset_z_mm"]),
            render_size=int(data["render_size"]),
            pixel_per_mm_at_sample=float(data["pixel_per_mm_at_sample"]),
            physical_distance_mm=float(data["physical_distance_mm"]),
        )
    return cfg
