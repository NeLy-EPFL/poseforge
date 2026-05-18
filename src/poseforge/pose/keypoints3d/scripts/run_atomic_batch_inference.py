#!/usr/bin/env python3
"""Run keypoints3d inference on pre-extracted atomic batches.

The script expects:
- an atomic batch directory containing paired `*_frames.mp4` and `*_labels.h5` files
- a model folder containing `configs/model_architecture_config.yaml`
- a full keypoints3d checkpoint in that model folder
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

from poseforge.pose.data.synthetic import AtomicBatchDataset


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file {path}, got {type(data).__name__}")
    return data


def _find_unique_file(base_dir: Path, patterns: tuple[str, ...], description: str) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(base_dir.rglob(pattern))
    unique_matches = sorted(set(matches))
    if len(unique_matches) == 0:
        raise FileNotFoundError(
            f"Could not find {description} under {base_dir}. Looked for: {list(patterns)}"
        )
    if len(unique_matches) > 1:
        raise ValueError(
            f"Multiple {description} files found under {base_dir}: {[str(path) for path in unique_matches]}. "
            "Pass an explicit override to disambiguate."
        )
    return unique_matches[0]


def _resolve_model_artifacts(
    model_dir: Path,
    *,
    architecture_config_path: Path | None = None,
    data_config_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    model_dir = Path(model_dir)

    if architecture_config_path is None:
        architecture_config_path = _find_unique_file(
            model_dir,
            ("configs/model_architecture_config.yaml", "model_architecture_config.yaml"),
            "model architecture config",
        )
    architecture_config = _load_yaml_dict(architecture_config_path)

    if data_config_path is None:
        try:
            data_config_path = _find_unique_file(
                model_dir,
                ("configs/data_config.yaml", "data_config.yaml"),
                "data config",
            )
        except FileNotFoundError:
            data_config_path = None

    data_config: dict[str, Any] = {}
    if data_config_path is not None:
        data_config = _load_yaml_dict(data_config_path)

    if checkpoint_path is None:
        try:
            checkpoint_path = _find_unique_file(
                model_dir,
                ("*.model.pth", "checkpoints/*.model.pth"),
                "model checkpoint",
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Could not find a full model checkpoint under {model_dir}. "
                "Provide --checkpoint explicitly if the model folder contains multiple candidates."
            ) from exc

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    return architecture_config_path, checkpoint_path, data_config_path, architecture_config | data_config


def _resolve_atomic_batch_settings(
    model_config: dict[str, Any],
    *,
    atomic_batch_n_variants: int | None,
    input_image_size: tuple[int, int] | None,
) -> tuple[int, tuple[int, int]]:
    inferred_n_variants = atomic_batch_n_variants
    inferred_image_size = input_image_size

    if inferred_n_variants is None:
        inferred_n_variants = model_config.get("atomic_batch_n_variants")
    if inferred_image_size is None:
        inferred_image_size = model_config.get("input_image_size")

    if inferred_n_variants is None:
        raise ValueError(
            "Could not determine atomic batch variant count. "
            "Pass --atomic-batch-n-variants or include it in the model data config."
        )
    if inferred_image_size is None:
        raise ValueError(
            "Could not determine atomic batch image size. "
            "Pass --input-image-size or include it in the model data config."
        )

    image_size_tuple = tuple(int(x) for x in inferred_image_size)
    if len(image_size_tuple) != 2:
        raise ValueError(
            f"input_image_size must contain exactly two values, got {inferred_image_size!r}"
        )

    return int(inferred_n_variants), image_size_tuple


    return architecture_config_path, checkpoint_path, data_config_path, architecture_config | data_config


def _build_keypoints3d_model_and_pipeline(
    architecture_config_path: Path,
    checkpoint_path: Path,
    device: torch.device | str,
):
    from poseforge.neuromechfly.constants import keypoint_segments_canonical
    from poseforge.pose.keypoints3d import Pose2p5DModel, Pose2p5DPipeline
    from poseforge.pose.keypoints3d import config as keypoints3d_config

    architecture_config = keypoints3d_config.ModelArchitectureConfig.load(
        architecture_config_path
    )
    if architecture_config.n_keypoints != len(keypoint_segments_canonical):
        raise ValueError(
            f"keypoints3d model expects {architecture_config.n_keypoints} keypoints, "
            f"but canonical keypoint list contains {len(keypoint_segments_canonical)} labels"
        )

    model = Pose2p5DModel.create_architecture_from_config(architecture_config)
    model.load_weights_from_config(
        keypoints3d_config.ModelWeightsConfig(model_weights=checkpoint_path)
    )
    pipeline = Pose2p5DPipeline(model, device=device, use_float16=True)
    return model, pipeline, list(keypoint_segments_canonical)


def _save_keypoints3d_predictions(
    output_path: Path,
    *,
    pred_xy: np.ndarray,
    pred_depth: np.ndarray,
    conf_xy: np.ndarray,
    conf_depth: np.ndarray,
    keypoint_names: list[str],
    metadata: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        xy_ds = f.create_dataset(
            "pred_xy",
            data=pred_xy,
            dtype=np.float16,
            compression="gzip",
        )
        xy_ds.attrs["keypoints"] = keypoint_names
        xy_ds.attrs["units"] = "pixels"

        depth_ds = f.create_dataset(
            "pred_depth",
            data=pred_depth,
            dtype=np.float16,
            compression="gzip",
        )
        depth_ds.attrs["keypoints"] = keypoint_names
        depth_ds.attrs["units"] = "mm"

        conf_xy_ds = f.create_dataset(
            "conf_xy",
            data=conf_xy,
            dtype=np.float16,
            compression="gzip",
        )
        conf_xy_ds.attrs["keypoints"] = keypoint_names

        conf_depth_ds = f.create_dataset(
            "conf_depth",
            data=conf_depth,
            dtype=np.float16,
            compression="gzip",
        )
        conf_depth_ds.attrs["keypoints"] = keypoint_names

        for key, value in metadata.items():
            f.attrs[key] = value


def run_atomic_batch_inference(
    atomic_batch_dir: Path,
    model_dir: Path,
    output_dir: Path | None = None,
    *,
    architecture_config_path: Path | None = None,
    data_config_path: Path | None = None,
    checkpoint_path: Path | None = None,
    atomic_batch_n_variants: int | None = None,
    input_image_size: tuple[int, int] | None = None,
    frames_serialization_spacing: int = 10,
    device: str = "cuda",
) -> None:
    atomic_batch_dir = Path(atomic_batch_dir)
    model_dir = Path(model_dir)

    if not atomic_batch_dir.is_dir():
        raise FileNotFoundError(f"Atomic batch directory does not exist: {atomic_batch_dir}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    architecture_config_path, checkpoint_path, resolved_data_config_path, model_config = _resolve_model_artifacts(
        model_dir,
        architecture_config_path=architecture_config_path,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
    )
    atomic_batch_n_variants_resolved, input_image_size_resolved = _resolve_atomic_batch_settings(
        model_config,
        atomic_batch_n_variants=atomic_batch_n_variants,
        input_image_size=input_image_size,
    )

    _, pipeline, keypoint_names = _build_keypoints3d_model_and_pipeline(
        architecture_config_path=architecture_config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    dataset = AtomicBatchDataset(
        data_dirs=[atomic_batch_dir],
        n_variants=atomic_batch_n_variants_resolved,
        image_size=input_image_size_resolved,
        n_channels=3,
        frames_serialization_spacing=frames_serialization_spacing,
        load_dof_angles=False,
        load_keypoint_positions=False,
        load_body_segment_maps=False,
    )

    checkpoint_label = checkpoint_path.name
    if checkpoint_label.endswith(".model.pth"):
        checkpoint_label = checkpoint_label[: -len(".model.pth")]
    elif checkpoint_label.endswith(".pth"):
        checkpoint_label = checkpoint_label[: -len(".pth")]

    if output_dir is None:
        output_dir = model_dir / "atomic_batch_inference" / "keypoints3d" / checkpoint_label
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Atomic batch directory: {atomic_batch_dir}")
    logging.info(f"Architecture config: {architecture_config_path}")
    logging.info(f"Data config: {resolved_data_config_path if resolved_data_config_path is not None else 'not found'}")
    logging.info(f"Checkpoint: {checkpoint_path}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Atomic batch variants: {atomic_batch_n_variants_resolved}")
    logging.info(f"Atomic batch image size: {input_image_size_resolved}")

    model_metadata = {
        "model_family": "keypoints3d",
        "architecture_config": str(architecture_config_path),
        "checkpoint_path": str(checkpoint_path),
        "data_config_path": str(resolved_data_config_path) if resolved_data_config_path is not None else "",
        "atomic_batch_n_variants": int(atomic_batch_n_variants_resolved),
        "input_image_size": list(input_image_size_resolved),
        "frames_serialization_spacing": int(frames_serialization_spacing),
    }

    for idx in tqdm(range(len(dataset)), desc="Atomic batches", unit="batch", disable=None):
        mp4_path, _ = dataset.atomic_batches[idx]
        frames, _ = dataset[idx]

        if frames.shape[0] != atomic_batch_n_variants_resolved:
            raise ValueError(
                f"Atomic batch {mp4_path} has {frames.shape[0]} variants after loading, "
                f"but the resolved model configuration expects {atomic_batch_n_variants_resolved}."
            )

        n_variants, n_frames, n_channels, n_rows, n_cols = frames.shape
        flat_frames = frames.reshape(n_variants * n_frames, n_channels, n_rows, n_cols)

        pred_dict = pipeline.inference(flat_frames)

        batch_root = output_dir / mp4_path.parent.relative_to(atomic_batch_dir)
        batch_root.mkdir(parents=True, exist_ok=True)
        batch_stem = mp4_path.name.removesuffix("_frames.mp4")

        pred_xy = pred_dict["pred_xy"].reshape(n_variants, n_frames, -1, 2)
        pred_depth = pred_dict["pred_depth"].reshape(n_variants, n_frames, -1)
        conf_xy = pred_dict["conf_xy"].reshape(n_variants, n_frames, -1)
        conf_depth = pred_dict["conf_depth"].reshape(n_variants, n_frames, -1)

        out_path = batch_root / f"{batch_stem}_keypoints3d_pred.h5"
        _save_keypoints3d_predictions(
            out_path,
            pred_xy=pred_xy.detach().cpu().numpy().astype(np.float16),
            pred_depth=pred_depth.detach().cpu().numpy().astype(np.float16),
            conf_xy=conf_xy.detach().cpu().numpy().astype(np.float16),
            conf_depth=conf_depth.detach().cpu().numpy().astype(np.float16),
            keypoint_names=keypoint_names,
            metadata=model_metadata,
        )
        logging.info(f"Wrote keypoints3d predictions to {out_path}")


def start() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on an atomic batch directory using a trained model folder."
    )
    parser.add_argument(
        "atomic_batch_dir",
        type=Path,
        help="Directory containing paired atomic batch files (`*_frames.mp4` and `*_labels.h5`).",
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing the model configs and checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write predictions. Defaults to <model_dir>/atomic_batch_inference/<family>/<checkpoint>.",
    )
    parser.add_argument(
        "--architecture-config",
        type=Path,
        default=None,
        help="Explicit path to model_architecture_config.yaml.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=None,
        help="Explicit path to data_config.yaml used to infer atomic batch settings.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit path to the full model checkpoint.",
    )
    parser.add_argument(
        "--atomic-batch-n-variants",
        type=int,
        default=None,
        help="Override the number of variants per atomic batch.",
    )
    parser.add_argument(
        "--input-image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Override the atomic batch frame size if it is not available from the model data config.",
    )
    parser.add_argument(
        "--frames-serialization-spacing",
        type=int,
        default=10,
        help="Horizontal spacing, in pixels, between serialized variants inside each atomic batch video.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to use for inference.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> None:
    args = start()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.input_image_size is not None:
        input_image_size = (int(args.input_image_size[0]), int(args.input_image_size[1]))
    else:
        input_image_size = None

    run_atomic_batch_inference(
        atomic_batch_dir=args.atomic_batch_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        architecture_config_path=args.architecture_config,
        data_config_path=args.data_config,
        checkpoint_path=args.checkpoint,
        atomic_batch_n_variants=args.atomic_batch_n_variants,
        input_image_size=input_image_size,
        frames_serialization_spacing=args.frames_serialization_spacing,
        device=args.device,
    )


if __name__ == "__main__":
    main()