import json
from pathlib import Path

import numpy as np
import torch

from poseforge.style_transfer import get_inference_pipeline, parse_hyperparameters_from_checkpoint_path
from poseforge.style_transfer.tiled_inference import process_simulation_tiled


def parse_video_filename_from_checkpoint_path(checkpoint_path: Path) -> str:
    """Infer expected simulation input filename from training dataroot."""
    train_opt_path = checkpoint_path.parent.parent / "train_options.json"
    if not train_opt_path.is_file():
        raise FileNotFoundError(
            f"Could not find train_options.json at expected path: {train_opt_path}"
        )

    with open(train_opt_path, "r") as f:
        dataroot = json.load(f)["values"]["dataroot"]

    if "flybody" in dataroot:
        if "grayscale" in dataroot or "gray" in dataroot:
            return "processed_nmf_sim_render_flybody_grayscale.mp4"
        if "base" in dataroot:
            return "processed_nmf_sim_render_flybody_base.mp4"
        if "link" in dataroot:
            return "processed_nmf_sim_render_per_link_color.mp4"
        if "segment_id" in dataroot or "segmentid" in dataroot:
            return "processed_nmf_sim_segment_id.mp4"
        if "depth" in dataroot:
            return "processed_nmf_sim_depth.mp4"
    else:
        if "gray" in dataroot or "grayscale" in dataroot:
            return "processed_nmf_sim_render_grayscale.mp4"
        if "base" in dataroot:
            return "processed_nmf_sim_render_base.mp4"
        if "link" in dataroot:
            return "processed_nmf_sim_render_per_link_color.mp4"
        if "segment_id" in dataroot or "segmentid" in dataroot:
            return "processed_nmf_sim_segment_id.mp4"
        if "depth" in dataroot:
            return "processed_nmf_sim_depth.mp4"
        if dataroot.endswith("aymanns2022_pseudocolor_spotlight_dataset"):
            return "processed_nmf_sim_render_per_link_color.mp4"

    raise ValueError(f"Could not parse video filename from dataroot: {dataroot}")


def _find_checkpoint_paths(checkpoint_root: Path, keyword: str = "patch", num_last_checkpoints: int | None = None) -> list[Path]:
    """Find all generator checkpoints under a root, keeping only patch runs."""
    keyword = keyword.lower()
    checkpoint_paths = []
    parent_dirs = []
    for path in checkpoint_root.rglob("[0-9]*_net_G.pth"):
        if any(keyword in part.lower() for part in path.parts):
            # parse checkpoint paths for every checkpoint folder
            parent_dir = path.parent
            if parent_dir not in parent_dirs:
                parent_dirs.append(parent_dir)
                checkpoint_paths.append([path])
            else:
                parent_dir_index = parent_dirs.index(parent_dir)
                checkpoint_paths[parent_dir_index].append(path)
    if num_last_checkpoints is not None:
        checkpoint_paths = [sorted(paths, key=lambda p: int(p.stem.split("_")[0]))[-num_last_checkpoints:] for paths in checkpoint_paths]
    checkpoint_paths = [path for sublist in checkpoint_paths for path in sublist]

    return sorted(checkpoint_paths, key=lambda path: (str(path.parent), int(path.stem.split("_")[0])))


def _build_output_dir(output_basedir: Path, checkpoint_root: Path, checkpoint_path: Path) -> Path:
    """Build a collision-free output directory for one checkpoint."""
    try:
        relative_parent = checkpoint_path.parent.relative_to(checkpoint_root)
        return output_basedir / relative_parent
    except ValueError:
        return output_basedir / checkpoint_path.parent.name


def run_tiled_inference_for_checkpoint(
    checkpoint_path: str,
    simulation_root: str,
    output_basedir: str | None = None,
    video_filename: str | None = None,
    patch_batch_size: int | None = None,
    weight_type: str = "uniform",
    debug_mode: bool = False,
    randomize_seams: bool = True,
    seed: int | None = 42,
    num_last_checkpoints: int | None = None,
    device: str = "cuda",
    progress_bar: bool = True,
    checkpoint_folder_keyword: str = "patch",
) -> None:
    """Run tiled style-transfer inference for one or more checkpoints.

    The full frame is split into minimally many, evenly distributed tiles of
    model input size. In overlapping regions, pixel values are merged with
    weighted averaging. If randomize_seams is True, seam locations vary
    per frame using random phases (deterministic if seed is provided).
    
    If num_last_checkpoints is specified, only the last N checkpoints from
    each folder will be processed; otherwise all matching checkpoints are used.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        device = "cpu"

    if output_basedir is None:
        base_root = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent.parent
        suffix = f"tiled_inference_w{weight_type}"
        if debug_mode:
            suffix += "_debug"
        if randomize_seams:
            suffix += "_random_seams"
        output_basedir = str((base_root / suffix).resolve())
    output_basedir = Path(output_basedir)

    if checkpoint_path.is_dir():
        checkpoint_paths = _find_checkpoint_paths(checkpoint_path, keyword=checkpoint_folder_keyword, num_last_checkpoints=num_last_checkpoints)
        checkpoint_root = checkpoint_path
    else:
        checkpoint_paths = [checkpoint_path]
        checkpoint_root = checkpoint_path.parent.parent

    if len(checkpoint_paths) == 0:
        print(
            f"No checkpoints found under {checkpoint_path} matching keyword '{checkpoint_folder_keyword}'."
        )
        return

    print(f"Found {len(checkpoint_paths)} checkpoint(s) to process.")
    
    # Create root RNG for deterministic but varying seams
    if randomize_seams and seed is not None:
        root_rng = np.random.RandomState(seed)
    else:
        root_rng = None
    
    simulation_root = Path(simulation_root)

    for checkpoint_file in checkpoint_paths:
        model_hparams = parse_hyperparameters_from_checkpoint_path(checkpoint_file)
        model_hparams["preprocess_opt"]["preprocess"] = ""
        inference_pipeline = get_inference_pipeline(checkpoint_file, model_hparams, device=device)

        is_flybody = "flybody" in str(checkpoint_file)
        # hardcode simulation data dirs for now
        simulation_root_spe = simulation_root / ("bulk_data_flybody" if is_flybody else "bulk_data")
        simulation_data_dirs = [
            simulation_root_spe / "BO_Gal4_fly5_trial005/segment_001/subsegment_000",
        ]
        simulation_data_dirs = [Path(x).expanduser().resolve() for x in simulation_data_dirs]

        if video_filename is None:
            checkpoint_video_filename = parse_video_filename_from_checkpoint_path(checkpoint_file)
        else:
            checkpoint_video_filename = video_filename

        try:
            epoch = int(checkpoint_file.stem.split("_")[0])
        except Exception:
            epoch = -1

        output_dir = _build_output_dir(output_basedir, checkpoint_root, checkpoint_file)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, sim_dir in enumerate(simulation_data_dirs):
            input_video_path = sim_dir / checkpoint_video_filename
            if not input_video_path.is_file():
                raise FileNotFoundError(f"Input video does not exist: {input_video_path}")

            epoch_tag = f"epoch{epoch:03d}" if epoch >= 0 else "epochXXX"
            output_name = f"{epoch_tag}_examplesim{i:02d}_tiled"
            if debug_mode:
                output_name += "_debug"
            output_video_path = output_dir / f"{output_name}.mp4"
            print(f"Processing {input_video_path} -> {output_video_path}")

            # Derive a unique seed for this video from the root RNG
            video_seed = None
            if root_rng is not None:
                video_seed = root_rng.randint(0, 2**31 - 1)

            process_simulation_tiled(
                inference_pipeline=inference_pipeline,
                input_video_path=input_video_path,
                output_video_path=output_video_path,
                randomize_seams=randomize_seams,
                seed=video_seed,
                patch_batch_size=patch_batch_size,
                weight_type=weight_type,
                debug_mode=debug_mode,
                progress_bar=progress_bar,
            )


if __name__ == "__main__":
    import tyro

    tyro.cli(run_tiled_inference_for_checkpoint)
