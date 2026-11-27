"""Extract the wall library + statistics from the real recordings (cluster data).

A frame is used as a wall source only when the fly's stage-derived distance to
the arena wall is in the configured band (default 1.9-2.5 mm: close enough that
the chamber edge is in view, far enough that it never overlaps the fly) and the
fly-removed residual holds a real border-touching structure. Writes:

- ``wall_library.npz``       (+ ``wall_library_preview.png``)
- ``wall_statistics.json``   (wall presence rate)

Example:
    python -m poseforge.dust.scripts.extract_walls \
        --dataset-root /path/to/261117_poseforgetraining \
        --arena-dir    /path/to/261117_poseforgetraining/arena72x48 \
        --output-dir   /work/.../poseforge/dust
"""

from pathlib import Path

from poseforge.dust.config import WallDistanceConfig, WallExtractionConfig
from poseforge.dust.stage import build_wall_library_from_trials, find_trials


def extract_walls(
    dataset_root: str,
    arena_dir: str,
    output_dir: str,
    wall_config: WallExtractionConfig = WallExtractionConfig(),
    distance_config: WallDistanceConfig = WallDistanceConfig(),
    max_frames_per_trial: int | None = None,
) -> None:
    """Build the wall library and its statistics, gated by stage distance.

    Args:
        dataset_root: Folder holding the trial subfolders (each with a
            ``processed/`` directory containing the aligned video, stage metadata
            and alignment transforms).
        arena_dir: The arena folder (e.g. ``arena72x48``) with ``metadata.yaml``
            and ``model/calibration_result.yaml``, used to turn stage positions
            into fly-to-wall distances.
        output_dir: Folder to write the library, preview and statistics into.
        wall_config: Wall residual / presence-check parameters.
        distance_config: Stage-distance band selecting wall-bearing frames.
        max_frames_per_trial: Optional cap on in-band frames read per trial.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = find_trials(dataset_root, distance_config)
    print(f"Found {len(trials)} trials under {dataset_root}")

    wall_library, statistics = build_wall_library_from_trials(
        trials, arena_dir, wall_config, distance_config, max_frames_per_trial=max_frames_per_trial
    )

    wall_library.save(output_dir / "wall_library.npz")
    wall_library.make_preview_sheet(output_dir / "wall_library_preview.png")
    statistics.save(output_dir / "wall_statistics.json")

    print(
        f"Wall library: {len(wall_library)} walls in "
        f"[{distance_config.min_distance_mm}, {distance_config.max_distance_mm}] mm"
    )
    print(f"Wall presence rate: {statistics.wall_presence_rate:.3f} (fraction of all frames)")
    print(f"Wrote wall_library.npz, preview and wall_statistics.json to {output_dir}")


if __name__ == "__main__":
    import tyro

    tyro.cli(extract_walls)
