#!/usr/bin/env python
"""Ground-truth self-consistency check for the inverse-kinematics pipeline.

Each pre-extracted atomic batch stores BOTH the IK *input* (``keypoint_pos`` =
per-keypoint [pixel_x, pixel_y, depth_mm]) AND the *answer* that generated it
(``dof_angles`` = the simulated joint angles, named via the dataset's ``keys``
attribute). That makes the IK directly testable: unproject the stored 2.5D
keypoints to 3D world coordinates, run the repo's IK, and check that the
recovered joint angles match the stored ground-truth angles.

This validates, end to end:
  * issue #48 / I3-A   - the restored ``dof_name_lookup_canonical_to_nmf`` lets
                          ``_save_seqikpy_output`` run and pack DOFs correctly;
  * issue #48 / I1-A   - the camera unprojection uses the correct sensor size
                          (the keypoint labels are in ``rendering_size`` px);
  * the IK solve itself (per-DOF angle error).

It also has a flygym-free ``--emit-bounds`` mode that regenerates the
data-derived ``nmf_bounds`` (issue #48 / I3-B) from the observed ``dof_angles``
range of motion.

Usage
-----
    # full self-consistency check (needs flygym + seqikpy installed):
    python scripts/verify_ik_selfconsistency.py \
        --data-dir bulk_data/pose_estimation/atomic_batches/4variants \
        --n-batches 5 --max-frames 32

    # regenerate data-derived joint bounds (no flygym needed):
    python scripts/verify_ik_selfconsistency.py \
        --data-dir bulk_data/pose_estimation/atomic_batches/4variants \
        --emit-bounds --n-batches 500

NOTE (implemented by Claude Code (Opus 4.8) upon user instruction; issue #48).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

# Camera unprojection only needs numpy/scipy, so it's safe to import eagerly.
from poseforge.pose.camera import CameraToWorldMapper

# Production camera (see production/spotlight/keypoints3d.py and
# run_keypoints3d_inference.py). The atomic-batch ``keypoint_pos`` x,y are in the
# stored frame's pixel space (256 px for the shipped 4variants data), so the
# mapper MUST be built at that size -- this is exactly the I1-A fix.
DEFAULT_CAMERA = dict(
    camera_pos=(0.0, 0.0, -100.0),
    camera_fov_deg=5.0,
    rotation_euler=(0.0, np.pi, -np.pi / 2.0),
)

# Atomic-batch keypoint part names (NMF-style) -> canonical leg-keypoint names
# that ``invkin.run_seqikpy`` / ``keypoint_segments_canonical`` expect.
PART_TO_CANONICAL = {
    "Coxa": "ThC",
    "Femur": "CTr",
    "Tibia": "FTi",
    "Tarsus1": "TiTa",
    "Tarsus5": "Claw",
}

# The 7 canonical DOFs per leg, in the order the IK output is packed.
CANONICAL_DOFS = [
    "ThC_yaw", "ThC_pitch", "ThC_roll",
    "CTr_pitch", "CTr_roll", "FTi_pitch", "TiTa_pitch",
]
LEGS = [f"{side}{pos}" for side in "LR" for pos in "FMH"]


def _decode_keys(attr) -> list[str]:
    return [k.decode() if isinstance(k, bytes) else str(k) for k in attr]


def keypoint_keys_to_canonical(data_keys: list[str]) -> list[str]:
    """Map atomic-batch keypoint names (e.g. ``LFCoxa``, ``LPedicel``) to the
    canonical names IK expects (e.g. ``LFThC``, ``LPedicel``)."""
    out = []
    for k in data_keys:
        if k.endswith("Pedicel"):  # antennae pass through unchanged
            out.append(k)
            continue
        leg, part = k[:2], k[2:]
        if part not in PART_TO_CANONICAL:
            raise ValueError(f"Unrecognized keypoint part {part!r} in {k!r}")
        out.append(f"{leg}{PART_TO_CANONICAL[part]}")
    return out


def circular_abs_diff_deg(a_rad: np.ndarray, b_rad: np.ndarray) -> np.ndarray:
    """|a - b| wrapped to [0, pi], in degrees (robust to angle wrapping)."""
    d = (a_rad - b_rad + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(np.abs(d))


def load_batches(data_dir: Path, n_batches: int, max_frames: int | None, seed: int):
    files = sorted(glob.glob(str(data_dir / "**" / "*_labels.h5"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No *_labels.h5 under {data_dir}")
    rng = np.random.default_rng(seed)
    pick = files if n_batches >= len(files) else [
        files[i] for i in rng.choice(len(files), n_batches, replace=False)
    ]
    import h5py

    kp_list, dof_list, kp_keys, dof_keys = [], [], None, None
    for f in pick:
        with h5py.File(f, "r") as h:
            kp = h["keypoint_pos"][:]
            dof = h["dof_angles"][:]
            if max_frames is not None:
                kp, dof = kp[:max_frames], dof[:max_frames]
            kp_list.append(kp)
            dof_list.append(dof)
            if kp_keys is None:
                kp_keys = _decode_keys(h["keypoint_pos"].attrs["keys"])
                dof_keys = _decode_keys(h["dof_angles"].attrs["keys"])
    return np.concatenate(kp_list, 0), np.concatenate(dof_list, 0), kp_keys, dof_keys


def emit_bounds(data_dir: Path, n_batches: int, margin_deg: float, seed: int) -> None:
    """Print a data-derived ``nmf_bounds`` dict (flygym-free). I3-B."""
    _, dof, _, dof_keys = load_batches(data_dir, n_batches, None, seed)
    A = np.degrees(dof)
    omin = {dof_keys[j]: A[:, j].min() for j in range(len(dof_keys))}
    omax = {dof_keys[j]: A[:, j].max() for j in range(len(dof_keys))}
    print(f"# data-derived from {A.shape[0]} frames, margin +/-{margin_deg:.0f} deg")
    print("nmf_bounds = {")
    for grp, pos in (("Front", "F"), ("Mid", "M"), ("Hind", "H")):
        print(f"    # {grp} legs")
        for leg in [f"{s}{pos}" for s in "RL"]:
            for dofn in CANONICAL_DOFS:
                col = f"{leg}{dofn}"
                if col not in omin:
                    continue
                lo = np.floor(omin[col] - margin_deg)
                hi = np.ceil(omax[col] + margin_deg)
                wrap = "  # WRAPPING: source RoM exceeds +/-180 deg" if (
                    omin[col] < -180 or omax[col] > 180) else ""
                print(f'    "{leg}_{dofn}": (np.deg2rad({lo:.0f}), np.deg2rad({hi:.0f})),{wrap}')
    print("}")


def run_selfconsistency(args) -> int:
    kp, gt_dof, kp_keys, dof_keys = load_batches(
        args.data_dir, args.n_batches, args.max_frames, args.seed
    )
    n_frames = kp.shape[0]
    print(f"Loaded {n_frames} frames; keypoints={len(kp_keys)} dofs={len(dof_keys)}")

    # 1) Unproject [px, py, depth] -> world mm with the CORRECT sensor size.
    mapper = CameraToWorldMapper(
        rendering_size=(args.rendering_size, args.rendering_size), **DEFAULT_CAMERA
    )
    world = mapper(kp[..., :2], kp[..., 2])  # (n_frames, n_kpts, 3) mm

    # 2) Run the repo's IK. Imported lazily so --emit-bounds works without flygym.
    import poseforge.pose.keypoints3d.invkin as invkin

    canonical_names = keypoint_keys_to_canonical(kp_keys)
    joint_angles, _fk = invkin.run_seqikpy(
        world_xyz=world,
        keypoint_names_canonical=canonical_names,
        n_workers=args.n_workers,
    )

    # 3) Compare recovered angles to ground truth, matching BY NAME (the IK output
    #    DOF order [yaw,pitch,roll,...] differs from the dof_angles order
    #    [pitch,roll,yaw,...], so positional comparison would be wrong).
    gt_index = {k: j for j, k in enumerate(dof_keys)}
    rows, all_err = [], []
    for leg in LEGS:
        for dofn in CANONICAL_DOFS:
            seqikpy_key = f"Angle_{leg}_{dofn}"
            gt_key = f"{leg}{dofn}"
            if seqikpy_key not in joint_angles or gt_key not in gt_index:
                print(f"  [skip] missing {seqikpy_key} or {gt_key}")
                continue
            rec = np.asarray(joint_angles[seqikpy_key]).reshape(-1)[:n_frames]
            gt = gt_dof[:, gt_index[gt_key]]
            err = circular_abs_diff_deg(rec, gt)
            rows.append((gt_key, float(err.mean()), float(np.percentile(err, 90))))
            all_err.append(err)

    all_err = np.concatenate(all_err)
    print(f"\n{'DOF':16s} {'MAE(deg)':>9s} {'p90(deg)':>9s}")
    for name, mae, p90 in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:16s} {mae:9.2f} {p90:9.2f}")
    print(f"\nOVERALL mean abs angle error: {all_err.mean():.2f} deg "
          f"(median {np.median(all_err):.2f}, p90 {np.percentile(all_err, 90):.2f})")
    print(f"fraction of DOF-frames within 5 deg: {(all_err < 5).mean() * 100:.1f}%")
    # Heuristic pass/fail: a correct mapping + camera + IK should sit well under
    # ~15 deg mean; gross mapping/ordering/camera bugs blow this up.
    threshold = args.threshold_deg
    ok = all_err.mean() < threshold
    print(f"\n{'PASS' if ok else 'FAIL'}: overall mean < {threshold:.0f} deg")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path,
                   default=Path("bulk_data/pose_estimation/atomic_batches/4variants"))
    p.add_argument("--n-batches", type=int, default=5,
                   help="number of atomic batches to sample")
    p.add_argument("--max-frames", type=int, default=32,
                   help="max frames per batch (None=all)")
    p.add_argument("--rendering-size", type=int, default=256,
                   help="pixel size of the stored keypoint labels (sensor size)")
    p.add_argument("--n-workers", type=int, default=6)
    p.add_argument("--threshold-deg", type=float, default=15.0,
                   help="pass if overall mean angle error is below this")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--emit-bounds", action="store_true",
                   help="print data-derived nmf_bounds and exit (no flygym needed)")
    p.add_argument("--margin-deg", type=float, default=10.0,
                   help="padding added to observed RoM when emitting bounds")
    args = p.parse_args()

    if args.emit_bounds:
        emit_bounds(args.data_dir, args.n_batches, args.margin_deg, args.seed)
        return 0
    return run_selfconsistency(args)


if __name__ == "__main__":
    raise SystemExit(main())
