"""Extract the dust library + statistics from the real recordings (cluster data).

Reads strided frames from every trial's aligned behaviour video and writes:

- ``dust_library.npz``       (+ ``dust_library_preview.png``)
- ``dust_statistics.json``   (presence rate, per-frame counts, sizes)

Example:
    python -m poseforge.dust.scripts.extract_dust \
        --dataset-root /path/to/261117_poseforgetraining \
        --output-dir   /work/.../poseforge/dust
"""

from pathlib import Path

from poseforge.dust.config import (
    DustExtractionConfig,
    LibraryBalanceConfig,
    WallDistanceConfig,
)
from poseforge.dust.stage import find_trials, iter_trial_frames
from poseforge.dust.statistics import build_dust_artifacts

# Trial layout (file names) is fixed for this dataset; only used to find trials
# and read the aligned video.
_LAYOUT = WallDistanceConfig()


def extract_dust(
    dataset_root: str,
    output_dir: str,
    dust_config: DustExtractionConfig = DustExtractionConfig(),
    balance: LibraryBalanceConfig = LibraryBalanceConfig(),
    frame_stride: int = 15,
    max_frames_per_trial: int | None = None,
) -> None:
    """Build the size-balanced dust library and its statistics from the trials.

    Args:
        dataset_root: Folder holding the trial subfolders (each with a
            ``processed/aligned_behavior_video.mkv``).
        output_dir: Folder to write the library, preview and statistics into.
        dust_config: Dust extraction parameters (edge exclusion, size filters...).
        balance: Size-balancing of the library (per-bin reservoir caps).
        frame_stride: Take every Nth aligned frame per trial.
        max_frames_per_trial: Optional cap on frames read per trial.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = find_trials(dataset_root, _LAYOUT)
    print(f"Found {len(trials)} trials under {dataset_root}")

    def frames():
        for trial in trials:
            yield from iter_trial_frames(
                trial, _LAYOUT, stride=frame_stride, max_frames=max_frames_per_trial
            )

    dust_library, bin_counts, statistics = build_dust_artifacts(frames(), dust_config, balance)

    dust_library.save(output_dir / "dust_library.npz")
    dust_library.make_preview_sheet(output_dir / "dust_library_preview.png")
    statistics.save(output_dir / "dust_statistics.json")

    counts = statistics.dust_counts_when_present
    mean_count = sum(counts) / len(counts) if counts else 0
    print(f"Frames measured: {statistics.n_frames_measured}")
    print(f"Dust library: {len(dust_library)} primitives (balanced). Bins: {bin_counts}")
    print(f"Dust presence rate: {statistics.dust_presence_rate:.3f} (fraction of frames)")
    print(f"Primitives per frame when present: mean {mean_count:.1f}")
    print(f"Wrote dust_library.npz, preview and dust_statistics.json to {output_dir}")


if __name__ == "__main__":
    import tyro

    tyro.cli(extract_dust)
