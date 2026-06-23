"""Tests for ``poseforge.pose.camera.CameraToWorldMapper``.

These tests pin down two facts that motivate the bug fix in
``run_keypoints3d_inference.py`` and ``production/spotlight/keypoints3d.py``:

1. The mapper is a correct inverse of its own forward projection (round-trip
   recovery to ~1e-6).
2. The mapper's intrinsics scale with the pixel resolution it is built with:
   the in-plane (x, y) focal scaling is proportional to the image size while
   depth is left untouched. Consequently, unprojecting predictions that live in
   ``inference_image_size`` (=256) pixels with a mapper built at 256 is correct,
   whereas building it at the simulation render size (=464) injects an
   anisotropic distortion (in-plane factor 256/464 ~= 0.55, depth factor 1.0).

``camera.py`` depends only on numpy and scipy, so these tests run standalone.
The module is loaded directly from the source file so the tests always exercise
the code under review regardless of where an editable install resolves
``poseforge`` to.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Load camera.py directly from this repo's source tree (sibling of tests/).
_CAMERA_PY = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "poseforge"
    / "pose"
    / "camera.py"
)
_spec = importlib.util.spec_from_file_location("_poseforge_camera_under_test", _CAMERA_PY)
_camera = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_camera)
CameraToWorldMapper = _camera.CameraToWorldMapper


# Camera parameters matching the inference/production defaults.
CAMERA_POS = (0.0, 0.0, -100.0)
CAMERA_FOV_DEG = 5.0
CAMERA_ROTATION_EULER = (0.0, np.pi, -np.pi / 2)

INFERENCE_SIZE = 256  # pixel space the model's pred_xy actually lives in
RENDERING_SIZE = 464  # simulation render size -- NOT the prediction space


def _make_mapper(size: int) -> "CameraToWorldMapper":
    return CameraToWorldMapper(
        CAMERA_POS, CAMERA_FOV_DEG, (size, size), CAMERA_ROTATION_EULER
    )


def _forward_project(mapper, world_xyz: np.ndarray):
    """Project world points to (pixel_xy, depth) with the mapper's own
    ``camera_matrix``. This is the exact inverse operation of ``__call__``.

    Args:
        world_xyz: (N, 3) world coordinates in mm.

    Returns:
        (pixel_xy (N, 2), depth (N,)).
    """
    hom = np.hstack([world_xyz, np.ones((world_xyz.shape[0], 1))])  # (N, 4)
    proj = (mapper.camera_matrix @ hom.T).T  # (N, 3): [x*z, y*z, z]
    depth = proj[:, 2]
    pixel_xy = proj[:, :2] / proj[:, 2:3]
    return pixel_xy, depth


def _to_camera_frame(mapper, world_xyz: np.ndarray) -> np.ndarray:
    """Map world coords into the mapper's camera frame. The camera-frame z is
    the depth; x, y are the in-plane coordinates."""
    hom = np.hstack([world_xyz, np.ones((world_xyz.shape[0], 1))])
    return (mapper.camera_extrinsic_mat @ hom.T).T[:, :3]


@pytest.mark.parametrize("size", [128, 256, 464])
def test_unprojection_inverts_projection(size):
    """Unprojecting the mapper's own forward projection recovers the original
    world points to numerical precision."""
    mapper = _make_mapper(size)

    # Several known world points spread around the camera target region (mm).
    world_xyz = np.array(
        [
            [0.5, -1.0, 0.2],
            [1.0, -2.0, 0.5],
            [-0.3, 0.4, -0.1],
            [2.0, 1.5, 1.0],
            [-1.2, -0.7, 0.9],
            [0.0, 0.0, 0.0],
        ]
    )

    pixel_xy, depth = _forward_project(mapper, world_xyz)
    recovered = mapper(pixel_xy, depth)

    assert recovered.shape == world_xyz.shape
    np.testing.assert_allclose(recovered, world_xyz, atol=1e-6, rtol=0)


def test_focal_scales_with_size_depth_does_not():
    """The in-plane focal scaling is linear in image size; depth is independent
    of image size.

    This is the root cause of the bug: the mapper's geometry depends on the
    pixel space it is built with, so it must be built at the resolution the
    predictions live in.
    """
    m256 = _make_mapper(INFERENCE_SIZE)
    m464 = _make_mapper(RENDERING_SIZE)

    focal_256 = -m256.focal_transform_mat[0, 0]
    focal_464 = -m464.focal_transform_mat[0, 0]

    # In-plane focal scaling is proportional to image size.
    np.testing.assert_allclose(
        focal_256 / focal_464, INFERENCE_SIZE / RENDERING_SIZE, rtol=1e-12
    )
    # ~0.55, NOT 1.0 -- so resolution matters for in-plane coords.
    assert abs(focal_256 / focal_464 - 0.55) < 0.01

    # The depth axis of the focal transform is the identity regardless of size.
    assert m256.focal_transform_mat[2, 2] == 1.0
    assert m464.focal_transform_mat[2, 2] == 1.0


def test_same_input_different_mapper_size_anisotropic_distortion():
    """Unprojecting the SAME (256-space) xy + depth with a 256-mapper vs a
    464-mapper differs, with in-plane scaling ~= 256/464 and depth scaling
    exactly 1.0.

    This confirms that 256 is the correct resolution for 256-space predictions:
    using a 464-mapper shrinks the recovered in-plane geometry by ~0.55 while
    leaving depth untouched -- precisely the anisotropy that corrupts the
    downstream inverse-kinematics joint angles.
    """
    m256 = _make_mapper(INFERENCE_SIZE)  # correct for 256-space pred_xy
    m464 = _make_mapper(RENDERING_SIZE)  # buggy: 256-space xy fed to a 464 mapper

    # pred_xy lives in 256-px space; depths are mm.
    xy = np.array(
        [
            [100.0, 120.0],
            [200.0, 50.0],
            [10.0, 240.0],
            [128.0, 128.0],
            [60.0, 200.0],
        ]
    )
    depth = np.array([95.0, 102.0, 100.0, 99.0, 98.0])

    world_256 = m256(xy, depth)
    world_464 = m464(xy, depth)

    # The two unprojections genuinely disagree (this is the bug's effect).
    assert not np.allclose(world_256, world_464)

    cam_256 = _to_camera_frame(m256, world_256)
    cam_464 = _to_camera_frame(m464, world_464)

    # Depth (camera-frame z) is exactly the input depth in BOTH cases: depth is
    # unaffected by the mapper resolution -> depth ratio is exactly 1.0.
    np.testing.assert_allclose(cam_256[:, 2], depth, atol=1e-9)
    np.testing.assert_allclose(cam_464[:, 2], depth, atol=1e-9)
    np.testing.assert_allclose(cam_256[:, 2] / cam_464[:, 2], 1.0, atol=1e-9)

    # In-plane geometry, however, is scaled. Hold the pixel displacement away
    # from each sensor's own principal point fixed (this isolates 1/focal): the
    # same +(dx, dy) pixel step maps to a camera-frame in-plane offset whose
    # magnitude scales as focal_464 / focal_256 = 464/256. Equivalently, the
    # 256-mapper's in-plane extent is 256/464 ~= 0.55x that of the 464-mapper
    # for the same predicted pixels.
    fixed_depth = np.array([100.0, 100.0])
    step = np.array([40.0, 25.0])  # arbitrary fixed pixel displacement

    def inplane_step_magnitude(mapper):
        pp = (mapper.rendering_size[0] - 1) / 2.0
        base_px = np.array([pp, pp])
        query_xy = np.stack([base_px, base_px + step])
        cam = _to_camera_frame(mapper, mapper(query_xy, fixed_depth))
        return np.linalg.norm(cam[1, :2] - cam[0, :2])

    inplane_256 = inplane_step_magnitude(m256)
    inplane_464 = inplane_step_magnitude(m464)

    # In-plane displacement ratio is the inverse of the focal ratio.
    np.testing.assert_allclose(
        inplane_256 / inplane_464, RENDERING_SIZE / INFERENCE_SIZE, rtol=1e-9
    )
    # Equivalently: the 464-mapper's in-plane response is 256/464 ~= 0.55x the
    # 256-mapper's, while depth response is identical (1.0x). Anisotropy.
    np.testing.assert_allclose(
        inplane_464 / inplane_256, INFERENCE_SIZE / RENDERING_SIZE, rtol=1e-9
    )
    assert abs(inplane_464 / inplane_256 - 0.55) < 0.01
