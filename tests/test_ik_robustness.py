"""Pure-logic tests for the inverse-kinematics robustness fixes (issue #48,
findings I3-A .. I3-E).

These tests deliberately avoid importing ``poseforge.neuromechfly.constants`` or
``poseforge.pose.keypoints3d.invkin`` directly: those modules import ``flygym``
(``from flygym.anatomy import JointDOF``), which is NOT installed in this
environment (it does not support Python 3.13 yet). Importing them would fail at
collection time.

Instead we:
  * parse the relevant source files with ``ast`` and assert on the literal
    definitions (the restored DOF map I3-A, and the mirror-consistency of
    ``nmf_bounds`` I3-B), and
  * extract the *actual source* of the pure numpy helpers added for I3-C / I3-D
    and ``exec`` them in an isolated namespace (numpy + a stub logger) so the
    tests exercise the real implementation rather than a replica.

NOTE (testing limitation): full IK import/run testing is blocked by the missing
``flygym`` dependency, so these tests cover the pure logic and the static
definitions only.
"""

import ast
import textwrap
from pathlib import Path

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# Locate source files relative to this test (repo-root/src/poseforge/...).
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTANTS_PY = REPO_ROOT / "src" / "poseforge" / "neuromechfly" / "constants.py"
INVKIN_PY = REPO_ROOT / "src" / "poseforge" / "pose" / "keypoints3d" / "invkin.py"
RUN_IK_PY = (
    REPO_ROOT
    / "src"
    / "poseforge"
    / "pose"
    / "keypoints3d"
    / "scripts"
    / "run_inverse_kinematics.py"
)


# --------------------------------------------------------------------------- #
# Helpers to read definitions out of the source without importing it.
# --------------------------------------------------------------------------- #
def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _eval_assigned_dict(module: ast.Module, name: str) -> dict:
    """Return the literally-assigned dict for a top-level ``name = {...}``.

    Evaluates value expressions in a tiny sandbox where ``np.deg2rad`` is the real
    numpy function (so bounds tuples come out in radians, exactly as the module
    would compute them) and ``np`` is numpy. Only used on trusted in-repo source.
    """
    for node in module.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if name in targets:
                return eval(  # noqa: S307 - trusted, in-repo source
                    compile(ast.Expression(node.value), "<ast-dict>", "eval"),
                    {"np": np, "__builtins__": {}},
                    {},
                )
    raise AssertionError(f"Top-level dict assignment {name!r} not found in source")


def _extract_function_source(path: Path, func_name: str) -> str:
    """Return the exact source text of a top-level function definition."""
    module = _parse_module(path)
    src = path.read_text()
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f"Function {func_name!r} not found in {path}")


def _load_pure_functions(path: Path, func_names: list[str]) -> dict:
    """Exec the given functions' real source in an isolated numpy-only namespace.

    A stub ``logger`` (loguru-like, swallows calls) is provided so the functions'
    logging calls do not require loguru. This lets us test the genuine helper
    bodies without importing the flygym-tainted module.
    """

    class _StubLogger:
        def __getattr__(self, _name):
            return lambda *a, **k: None

    ns = {"np": np, "logger": _StubLogger()}
    for fn in func_names:
        exec(_extract_function_source(path, fn), ns)  # noqa: S102 - trusted source
    return ns


# --------------------------------------------------------------------------- #
# I3-A: the deleted DOF map must be restored, with the right keys/order/values.
# --------------------------------------------------------------------------- #
# seqikpy emits joint-angle dict keys "Angle_{leg}_{canonical_dof}" in this order
# (see seqikpy.leg_inverse_kinematics.LegInvKinSeq); the IK save path packs DOFs
# in the order of dof_name_lookup_canonical_to_nmf.keys(), so the keys must be
# exactly these canonical names in this order.
EXPECTED_CANONICAL_DOFS_IN_ORDER = [
    "ThC_yaw",
    "ThC_pitch",
    "ThC_roll",
    "CTr_pitch",
    "CTr_roll",
    "FTi_pitch",
    "TiTa_pitch",
]
# The canonical -> NMF DOF name mapping (flygym-v1 NMF joint names).
EXPECTED_CANONICAL_TO_NMF = {
    "ThC_yaw": "Coxa_yaw",
    "ThC_pitch": "Coxa",
    "ThC_roll": "Coxa_roll",
    "CTr_pitch": "Femur",
    "CTr_roll": "Femur_roll",
    "FTi_pitch": "Tibia",
    "TiTa_pitch": "Tarsus1",
}


def test_i3a_dof_map_restored_with_correct_keys_and_order():
    module = _parse_module(CONSTANTS_PY)
    dof_map = _eval_assigned_dict(module, "dof_name_lookup_canonical_to_nmf")

    # Keys must be exactly the 7 canonical DOFs, in the documented order.
    assert list(dof_map.keys()) == EXPECTED_CANONICAL_DOFS_IN_ORDER
    # Values must be the matching NMF DOF names.
    assert dof_map == EXPECTED_CANONICAL_TO_NMF


def test_i3a_dof_keys_match_seqikpy_emitted_angle_keys():
    """The call sites build `Angle_{leg}_{dof}` from these keys; every key must be
    a canonical DOF name that seqikpy actually emits (ThC_yaw, ..., TiTa_pitch)."""
    module = _parse_module(CONSTANTS_PY)
    dof_map = _eval_assigned_dict(module, "dof_name_lookup_canonical_to_nmf")
    assert set(dof_map.keys()) == set(EXPECTED_CANONICAL_DOFS_IN_ORDER)
    # 7 leg DOFs total (matches the (n_frames, 6, 7) joint-angle array).
    assert len(dof_map) == 7


def test_i3a_call_sites_consume_the_restored_constant():
    """Guard against the call sites being changed out from under the constant."""
    for path in (RUN_IK_PY, REPO_ROOT
                 / "src" / "poseforge" / "production" / "spotlight" / "keypoints3d.py"):
        text = path.read_text()
        assert "dof_name_lookup_canonical_to_nmf.keys()" in text, (
            f"{path} no longer consumes dof_name_lookup_canonical_to_nmf; "
            f"update this test and the constant together."
        )


# --------------------------------------------------------------------------- #
# I3-B: nmf_bounds must be L/R mirror-consistent for every leg/DOF.
# --------------------------------------------------------------------------- #
LEG_DOFS = [
    "ThC_yaw",
    "ThC_pitch",
    "ThC_roll",
    "CTr_pitch",
    "CTr_roll",
    "FTi_pitch",
    "TiTa_pitch",
]


def _mirror_of_left(dof: str, lo: float, hi: float) -> tuple[float, float]:
    """Expected RIGHT bound given the LEFT bound (lo, hi).

    Convention (see constants.py I3-B comment): roll/yaw mirror by negating and
    swapping the limits -> R = (-hi, -lo); pitch is shared -> R = (lo, hi).
    """
    if dof.endswith("pitch"):
        return (lo, hi)
    return (-hi, -lo)


@pytest.fixture(scope="module")
def nmf_bounds() -> dict:
    module = _parse_module(CONSTANTS_PY)
    return _eval_assigned_dict(module, "nmf_bounds")


def test_i3b_all_legs_dofs_present(nmf_bounds):
    for side in "LR":
        for pos in "FMH":
            for dof in LEG_DOFS:
                assert f"{side}{pos}_{dof}" in nmf_bounds


@pytest.mark.parametrize("pos", ["F", "M", "H"])
@pytest.mark.parametrize("dof", LEG_DOFS)
def test_i3b_left_right_bounds_are_mirror_consistent(nmf_bounds, pos, dof):
    left = tuple(nmf_bounds[f"L{pos}_{dof}"])
    right = tuple(nmf_bounds[f"R{pos}_{dof}"])
    expected_right = _mirror_of_left(dof, *left)
    assert right == pytest.approx(expected_right), (
        f"{pos}_{dof}: right bound {np.rad2deg(right)} deg is not the mirror of "
        f"left bound {np.rad2deg(left)} deg (expected "
        f"{np.rad2deg(expected_right)} deg)."
    )


def test_i3b_no_physically_implausible_lower_bound(nmf_bounds):
    """The old RF/RM CTr_pitch lower bound was -270 deg (implausible). After the
    fix, no bound should exceed +/-180 deg."""
    for key, (lo, hi) in nmf_bounds.items():
        assert np.rad2deg(lo) >= -180.0 - 1e-6, f"{key} lower bound < -180 deg"
        assert np.rad2deg(hi) <= 180.0 + 1e-6, f"{key} upper bound > 180 deg"


def test_i3b_specific_fixed_values(nmf_bounds):
    """Pin the exact post-fix values for the five corrected bounds."""
    expected_deg = {
        "RF_ThC_roll": (-90, 10),
        "RF_CTr_pitch": (-180, 10),
        "RM_ThC_yaw": (-90, 45),
        "RM_CTr_pitch": (-180, 10),
        "RH_ThC_yaw": (-90, 45),
    }
    for key, (lo_deg, hi_deg) in expected_deg.items():
        lo, hi = nmf_bounds[key]
        assert (np.rad2deg(lo), np.rad2deg(hi)) == pytest.approx((lo_deg, hi_deg))


# --------------------------------------------------------------------------- #
# I3-D: NaN-frame interpolation helper (real source, numpy-only sandbox).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def interp_fn():
    ns = _load_pure_functions(INVKIN_PY, ["_interpolate_nan_frames"])
    return ns["_interpolate_nan_frames"]


def test_i3d_no_nan_is_passthrough(interp_fn):
    data = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    filled, n_nan, n_frames = interp_fn(data)
    assert n_nan == 0
    assert n_frames == 0
    assert np.array_equal(filled, data)


def test_i3d_interior_gap_is_linearly_interpolated(interp_fn):
    # One keypoint, one coord effectively: build (n_frames, 1, 1)-ish via 3 frames.
    data = np.zeros((3, 1, 3), dtype=np.float32)
    data[:, 0, 0] = [0.0, np.nan, 4.0]  # interior NaN -> should become 2.0
    data[:, 0, 1] = [1.0, 1.0, 1.0]
    data[:, 0, 2] = [0.0, 2.0, 4.0]
    filled, n_nan, n_affected = interp_fn(data)
    assert n_nan == 1
    assert n_affected == 1
    assert filled[1, 0, 0] == pytest.approx(2.0)
    assert not np.isnan(filled).any()


def test_i3d_edge_nans_are_filled_with_nearest(interp_fn):
    data = np.zeros((4, 1, 3), dtype=np.float32)
    # leading and trailing NaN -> forward/backward fill at edges
    data[:, 0, 0] = [np.nan, 2.0, 4.0, np.nan]
    filled, n_nan, n_affected = interp_fn(data)
    assert n_nan == 2
    assert n_affected == 2
    assert filled[0, 0, 0] == pytest.approx(2.0)  # backfilled from first valid
    assert filled[3, 0, 0] == pytest.approx(4.0)  # forward-filled from last valid
    assert not np.isnan(filled).any()


def test_i3d_fully_nan_series_stays_nan(interp_fn):
    data = np.zeros((3, 1, 3), dtype=np.float32)
    data[:, 0, 0] = [np.nan, np.nan, np.nan]  # whole series NaN -> unrecoverable
    data[:, 0, 1] = [1.0, 2.0, 3.0]
    filled, n_nan, n_affected = interp_fn(data)
    assert n_nan == 3
    assert n_affected == 3
    # The fully-NaN coordinate cannot be recovered and must remain NaN so the
    # caller can detect and raise.
    assert np.isnan(filled[:, 0, 0]).all()
    # The healthy coordinate is untouched.
    assert np.array_equal(filled[:, 0, 1], np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_i3d_affected_frame_count(interp_fn):
    data = np.zeros((4, 2, 3), dtype=np.float32)
    data[1, 0, 0] = np.nan
    data[1, 1, 2] = np.nan  # same frame, two NaNs
    data[3, 0, 1] = np.nan  # another frame
    _, n_nan, n_affected = interp_fn(data)
    assert n_nan == 3
    assert n_affected == 2  # frames 1 and 3


# --------------------------------------------------------------------------- #
# I3-C: large frame-to-frame joint-angle jump detector (real source).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def jump_fns():
    return _load_pure_functions(
        INVKIN_PY,
        ["detect_large_joint_angle_jumps", "log_large_joint_angle_jumps"],
    )


def test_i3c_detects_jump_above_threshold(jump_fns):
    detect = jump_fns["detect_large_joint_angle_jumps"]
    series = np.array([0.0, 0.1, 0.2, 2.0, 2.1])  # big jump 0.2 -> 2.0
    masks = detect({"Angle_LF_ThC_yaw": series}, threshold_rad=1.0)
    mask = masks["Angle_LF_ThC_yaw"]
    assert mask.dtype == bool
    assert mask[0] == False  # frame 0 never flagged
    assert mask[3] == True  # the 1.8 rad jump
    assert mask.sum() == 1


def test_i3c_no_false_positive_for_smooth_series(jump_fns):
    detect = jump_fns["detect_large_joint_angle_jumps"]
    series = np.linspace(0.0, 1.0, 50)  # max step ~0.02 rad
    masks = detect({"Angle_RF_FTi_pitch": series}, threshold_rad=np.deg2rad(45))
    assert masks["Angle_RF_FTi_pitch"].sum() == 0


def test_i3c_skips_non_1d_and_short_series(jump_fns):
    detect = jump_fns["detect_large_joint_angle_jumps"]
    masks = detect(
        {
            "scalar": np.array([1.0]),  # too short
            "twod": np.zeros((3, 2)),  # not 1D
            "ok": np.array([0.0, 10.0]),  # valid, one jump
        },
        threshold_rad=1.0,
    )
    assert "scalar" not in masks
    assert "twod" not in masks
    assert masks["ok"].tolist() == [False, True]


def test_i3c_log_helper_counts_total_jumps(jump_fns):
    log = jump_fns["log_large_joint_angle_jumps"]
    total = log(
        {
            "Angle_LF_ThC_yaw": np.array([0.0, 0.0, 5.0]),  # 1 jump
            "Angle_LF_ThC_pitch": np.array([0.0, 5.0, 10.0]),  # 2 jumps
        },
        threshold_rad=1.0,
    )
    assert total == 3


def test_i3c_log_helper_zero_when_smooth(jump_fns):
    log = jump_fns["log_large_joint_angle_jumps"]
    total = log({"a": np.linspace(0, 1, 100)}, threshold_rad=np.deg2rad(45))
    assert total == 0
