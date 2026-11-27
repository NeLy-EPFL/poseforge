"""Add dust to all style-transferred simulation videos, into a mirrored folder tree.

This is the post-CUT step: the style transfer produces clean translated videos;
this script discovers every translated video under ``input_basedir``, additively
composites a random amount of dust onto each frame, and writes the result to a
directory tree under ``output_basedir`` that mirrors the input layout.

It deliberately mirrors ``run_tiled_inference_all_simulations.py`` (discovery,
mirrored output tree, ``no_override``, ``simulation_name_glob``) so it slots into
the same cluster workflow.

Example:
    python scripts_on_cluster/add_dust/add_dust_all_simulations.py \
        --input-basedir  /work/.../tiled_translated_videos \
        --output-basedir /work/.../tiled_translated_videos_dusty \
        --dust-library-path /work/.../dust_library.npz
"""

import fnmatch
import logging
from pathlib import Path

from tqdm import tqdm
from pvio.io import read_frames_from_video, write_frames_to_video

from poseforge.dust.compositing import add_artifacts_to_frames
from poseforge.dust.config import (
    DustCompositingConfig,
    WallCompositingConfig,
)
from poseforge.dust.primitives import DustLibrary
from poseforge.dust.statistics import DustStatistics, WallStatistics
from poseforge.dust.walls import WallLibrary


def find_all_translated_videos(input_basedir: Path, video_glob: str) -> list[Path]:
    """Find every translated video under the input base directory."""
    return sorted(set(input_basedir.rglob(video_glob)))


def add_dust_all_simulations(
    input_basedir: str,
    output_basedir: str,
    dust_library_path: str,
    dust_statistics_path: str | None = None,
    wall_library_path: str | None = None,
    wall_statistics_path: str | None = None,
    video_glob: str = "translated_*.mp4",
    config: DustCompositingConfig = DustCompositingConfig(),
    wall_config: WallCompositingConfig = WallCompositingConfig(),
    seed: int = 0,
    no_override: bool = False,
    simulation_name_glob: str | None = None,
    verbose: bool = False,
) -> None:
    """Add dust and walls to every translated video, mirroring the tree.

    Dust and walls are applied independently, each at its own measured presence
    probability (from the statistics files, overridable via the config knobs). The
    dust amount and sizes come from the measured distribution by default. Pass
    only the dust library (no wall library) to add dust alone.

    Args:
        input_basedir: Base directory of clean, style-transferred videos.
        output_basedir: Base directory for the dusty videos (mirrors the input).
        dust_library_path: ``.npz`` dust library from ``extract_dust``.
        dust_statistics_path: ``dust_statistics.json`` driving dust presence/amount.
        wall_library_path: Optional ``.npz`` wall library from ``extract_walls``.
        wall_statistics_path: ``wall_statistics.json`` driving wall presence.
        video_glob: Glob matching translated videos within the input tree.
        config: Dust compositing parameters / manual overrides.
        wall_config: Wall compositing parameters (intensity/blur, probability override).
        seed: Base random seed; combined with each video and frame index.
        no_override: If True, skip videos whose dusty output already exists.
        simulation_name_glob: Restrict to videos whose first path component matches.
        verbose: Whether to print detailed logs.
    """
    input_basedir = Path(input_basedir)
    output_basedir = Path(output_basedir)

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    print(f"Loading dust library from {dust_library_path} ...")
    library = DustLibrary.load(dust_library_path)
    primitives = library.primitives
    print(f"Loaded {len(primitives)} dust primitives.")
    if len(primitives) == 0:
        print("Dust library is empty; nothing to composite.")
        return

    dust_statistics = None
    if dust_statistics_path is not None:
        dust_statistics = DustStatistics.load(dust_statistics_path)
        print(f"Dust presence rate: {dust_statistics.dust_presence_rate:.3f}")

    walls = []
    wall_statistics = None
    if wall_library_path is not None:
        walls = WallLibrary.load(wall_library_path).walls
        print(f"Loaded {len(walls)} wall primitives.")
        if wall_statistics_path is not None:
            wall_statistics = WallStatistics.load(wall_statistics_path)
            print(f"Wall presence rate: {wall_statistics.wall_presence_rate:.3f}")

    video_paths = find_all_translated_videos(input_basedir, video_glob)
    if simulation_name_glob:
        video_paths = [
            p
            for p in video_paths
            if fnmatch.fnmatch(
                p.relative_to(input_basedir).parts[0], simulation_name_glob
            )
        ]
        print(
            f"Filtered by glob {simulation_name_glob!r}: {len(video_paths)} videos remain"
        )
    print(f"Total number of videos to process: {len(video_paths)}")
    if len(video_paths) == 0:
        print(f"No videos matching {video_glob!r} found under {input_basedir}")
        return

    for video_path in tqdm(video_paths, disable=None):
        output_path = Path(
            str(video_path).replace(str(input_basedir), str(output_basedir))
        )
        assert (
            output_basedir in output_path.parents
        ), "Output path is outside the specified output base directory"

        if no_override and output_path.is_file():
            logging.info(f"Output already exists, skipping: {output_path}")
            continue

        try:
            frames, fps = read_frames_from_video(video_path)
            video_key = str(video_path.relative_to(input_basedir))
            dusty_frames = add_artifacts_to_frames(
                frames, primitives, walls, config, wall_config,
                dust_statistics, wall_statistics, seed=seed, video_key=video_key,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_frames_to_video(output_path, dusty_frames, fps=fps)
        except Exception as error:
            logging.error(f"Failed to process {video_path}: {error}")
            continue

    print(f"Finished processing {len(video_paths)} videos.")


if __name__ == "__main__":
    import tyro

    tyro.cli(add_dust_all_simulations)
