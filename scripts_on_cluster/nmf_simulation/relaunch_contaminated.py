#!/usr/bin/env python3
"""Detect *contaminated* NMF simulations and regenerate batch scripts to
re-simulate ONLY the affected (trial, segment) pairs.

This is the data-driven sibling of ``relaunch_oom.py``. Instead of reading the
SLURM logs for OOM kills, it reads the postprocessed keypoint depth from each
``processed_simulation_data.h5`` and flags any subsegment whose median depth is
inconsistent with the camera calibration. The canonical failure mode is stale
pre-recalibration data (camera ~-67 mm instead of the calibrated ~-138 mm) left
behind as an orphaned subsegment by an earlier, larger run -- see
``postprocess_segment`` (which now clears stale subsegments before writing).

Re-running the generated scripts re-simulates and re-postprocesses the flagged
segments; the subsegment-cleanup guard in ``postprocess_segment`` removes the
orphan so the segment comes back clean.

IMPORTANT: the same orphan also lives in the *style-transfer* output tree (the
``translated_*.mp4`` that atomic-batch extraction globs). Re-simulating fixes
the rendering tree but NOT the style-transfer tree, so after these jobs finish
you must also re-run style transfer for the affected trials (and re-extract) --
otherwise extraction will KeyError on a subsegment whose label no longer exists.
The contaminated subsegment paths are printed at the end for exactly that
cleanup.

Usage:
    python relaunch_contaminated.py relaunch_contaminated
    ./submit_all.sh relaunch_contaminated
"""

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).parent
TEMPLATE_PATH = SCRIPT_DIR / "template.run"

# Mirror the paths defined in gen_batch_scripts.py so the regenerated jobs land
# in exactly the same place as the original generation.
DATA_DIR = Path("/work/upramdya/stimpfli/poseforge")
RECORDED_TRIALS_DIR = DATA_DIR / "data/aymanns2022/trials"
USE_FLYBODY = "flybody" in str(RECORDED_TRIALS_DIR)
BIOMECH_MODEL = "flybody" if USE_FLYBODY else "nmf"
OUTPUT_BASEDIR = DATA_DIR / f"data/{BIOMECH_MODEL}_rendering"
LOG_DIR = SCRIPT_DIR / f"logs_{BIOMECH_MODEL}"

DEPTH_DATASET = "postprocessed/keypoint_pos/camera_coords"


def find_contaminated_segments(center: float, tol: float):
    """Scan every processed_simulation_data.h5 and return
    ({trial: sorted([segment_id, ...])}, [contaminated_subsegment_relpath, ...]).

    A subsegment is contaminated if the median of its keypoint depths
    (camera_coords[..., 2]) deviates from ``center`` by more than ``tol`` mm.
    """
    contaminated_segs: dict[str, set] = {}
    contaminated_subseg_paths: list[str] = []
    procs = sorted(
        glob.glob(str(OUTPUT_BASEDIR / "**" / "processed_simulation_data.h5"), recursive=True)
    )
    n_bad = 0
    for proc in procs:
        try:
            with h5py.File(proc, "r") as h:
                depth = h[DEPTH_DATASET][..., 2]
        except Exception as exc:
            print(f"  WARNING: could not read {proc}: {exc}")
            continue
        median = float(np.median(depth))
        if abs(median - center) > tol:
            n_bad += 1
            rel = Path(proc).relative_to(OUTPUT_BASEDIR)
            trial = rel.parts[0]
            segment_id = int(rel.parts[1].split("_")[1])  # "segment_006" -> 6
            contaminated_segs.setdefault(trial, set()).add(segment_id)
            contaminated_subseg_paths.append(str(rel.parent))  # trial/segment/subsegment
            print(f"  CONTAMINATED  depth~{median:8.1f} mm   {rel.parent}")
    print(
        f"\nScanned {len(procs)} subsegments; {n_bad} contaminated "
        f"across {len(contaminated_segs)} trial(s)."
    )
    return (
        {trial: sorted(ids) for trial, ids in contaminated_segs.items()},
        sorted(contaminated_subseg_paths),
    )


def make_run_script(trial: str, segment_ids: list[int], template_str: str, output_dir: Path) -> Path:
    segment_ids_str = [str(i) for i in segment_ids]
    job_name = f"{trial}_segs{'-'.join(segment_ids_str)}"
    log_output_file = LOG_DIR / f"{job_name}.out"
    recorded_trial_path = RECORDED_TRIALS_DIR / f"{trial}.pkl"
    trial_output_dir = OUTPUT_BASEDIR / trial
    use_flybody_flag = "--use-flybody" if USE_FLYBODY else "--no-use-flybody"
    if not recorded_trial_path.is_file():
        print(f"  WARNING: kinematic recording not found: {recorded_trial_path}")
    script_str = (
        template_str
        .replace("<<<JOB_NAME>>>", job_name)
        .replace("<<<LOG_OUTPUT_FILE>>>", str(log_output_file))
        .replace("<<<RECORDED_TRIAL_PATH>>>", str(recorded_trial_path))
        .replace("<<<TRIAL_OUTPUT_DIR>>>", str(trial_output_dir))
        .replace("<<<SEGMENT_IDS>>>", " ".join(segment_ids_str))
        .replace("<<<USE_FLYBODY_FLAG>>>", use_flybody_flag)
        .replace("<<<BIOMECH_MODEL>>>", BIOMECH_MODEL)
    )
    dest = output_dir / f"{job_name}.run"
    dest.write_text(script_str)
    return dest


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate batch scripts to re-simulate NMF segments whose "
            "postprocessed keypoint depth is inconsistent with the camera "
            "calibration (e.g. stale -67 mm orphans). Mirrors relaunch_oom.py."
        )
    )
    parser.add_argument(
        "output_folder",
        help=(
            "Destination folder for regenerated scripts (relative to this "
            "script's directory, e.g. 'relaunch_contaminated')."
        ),
    )
    parser.add_argument(
        "--valid-depth-center",
        type=float,
        default=-138.2,
        help=(
            "Expected median keypoint depth (mm) from the camera calibration "
            "(spotlight_cam_mat.npz pos_offset_z_mm). Default -138.2."
        ),
    )
    parser.add_argument(
        "--valid-depth-tol",
        type=float,
        default=25.0,
        help=(
            "Allowed deviation (mm) from the center before a subsegment is "
            "flagged contaminated. Default 25 (passes ~-138, flags ~-67)."
        ),
    )
    args = parser.parse_args()

    template_str = TEMPLATE_PATH.read_text()
    print(f"Using template: {TEMPLATE_PATH}")
    print(f"Scanning: {OUTPUT_BASEDIR}")
    print(f"Valid depth band: {args.valid_depth_center} +/- {args.valid_depth_tol} mm\n")

    contaminated_segs, contaminated_subseg_paths = find_contaminated_segments(
        args.valid_depth_center, args.valid_depth_tol
    )
    if not contaminated_segs:
        print("No contaminated segments found.")
        return

    output_dir = SCRIPT_DIR / args.output_folder
    if output_dir.exists() and list(output_dir.glob("*.run")):
        raise RuntimeError(
            f"{output_dir} already contains .run files. "
            "Empty it manually to avoid mixing stale scripts with retries."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    n_scripts = 0
    for trial in sorted(contaminated_segs):
        dest = make_run_script(trial, contaminated_segs[trial], template_str, output_dir)
        print(f"  Generated: {dest.name}  (segments {contaminated_segs[trial]})")
        n_scripts += 1

    print(f"\n{n_scripts} script(s) written to {output_dir}")
    print(f"Submit with:  ./submit_all.sh {args.output_folder}")
    print(
        "\nAFTER these jobs finish, the SAME subsegment paths must be cleaned "
        "from the style-transfer tree and the dataset re-extracted, or "
        "extraction will KeyError. Contaminated subsegments:"
    )
    for rel in contaminated_subseg_paths:
        print(f"  {rel}")


if __name__ == "__main__":
    main()
