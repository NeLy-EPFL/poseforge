"""Compute the within-/between-frame variant-similarity heatmaps and the
aggregate invariance ratio for trained feature extractor checkpoints.

Method-agnostic: works for any checkpoint saved by the contrastive
pipeline or the SimSiam pipeline (both write `feature_extractor.pth` in the
same format). The script:

1. Builds a ResNetFeatureExtractor and loads the given weights.
2. Iterates atomic batches in the given data dirs, applies the same
   aligned center-crop logic used at validation time.
3. Pools features the same way the training pipeline does.
4. Calls the shared compute_alignment_metrics helper (which mean-centers the
   features to defeat anisotropy) and accumulates its within-frame and
   between-frame variant-similarity matrices across batches.
5. Renders both heatmaps to PNGs and reports the aggregate invariance ratio.

Two subcommands:

- ``single``: evaluate one checkpoint (an explicit ``feature_extractor.pth``
  path) and write its within/between heatmaps + summary. Use this to inspect
  a finished run's variant similarity outside of TensorBoard.

- ``compare``: evaluate MANY training trials head-to-head on the *same*
  held-out data with the *same* code path, then rank them. Point it at a list
  of trial output directories (each containing ``checkpoints/`` and
  ``configs/``). For each trial it scores *every* checkpoint on the shared eval
  set and keeps that trial's best-scoring one (lowest invariance ratio), so a
  run is judged at its peak rather than at whatever its final step happened to
  be; it then ranks the per-trial bests against one another and writes a
  comparison table (CSV + text) plus a bar chart. This is the cluster
  entrypoint for benchmarking e.g. contrastive vs SimSiam, or a hyperparameter
  sweep, fairly against one another. Lower invariance ratio = more
  style-invariant features = better.
"""

import logging
import re
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from poseforge.pose.common import ResNetFeatureExtractor
from poseforge.pose.data.synthetic import (
    init_atomic_dataset_and_dataloader,
    concat_atomic_batches,
    collapse_batch,
)
from poseforge.pose.contrast.model import compute_alignment_metrics
from poseforge.pose.contrast.pipeline import _render_variant_similarity_heatmap


# Checkpoints are written by the training pipelines as
# ``checkpoint_epoch{E:03d}_step{S:06d}.feature_extractor.pth``.
_CHECKPOINT_RE = re.compile(
    r"^checkpoint_epoch(\d+)_step(\d+)\.feature_extractor\.pth$"
)
# The ``epochE_stepS`` stage identifier embedded in checkpoint names.
_STAGE_RE = re.compile(r"epoch(\d+)_step(\d+)")


def _parse_stage(stage: str) -> tuple[int | None, int | None]:
    """Parse ``(epoch, step)`` from an ``epochE_stepS`` stage string."""
    match = _STAGE_RE.search(stage)
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


# --------------------------------------------------------------------------- #
# Checkpoint and trial-metadata discovery
# --------------------------------------------------------------------------- #
def _find_checkpoints(trial_dir: str | Path) -> list[tuple[tuple[int, int], str, Path]]:
    """Return ``[((epoch, step), stage_str, path), ...]`` sorted ascending.

    ``stage_str`` is the ``epochE_stepS`` identifier used by the training
    pipelines, and ``path`` points at the ``feature_extractor.pth`` file.
    """
    checkpoint_dir = Path(trial_dir) / "checkpoints"
    if not checkpoint_dir.is_dir():
        return []
    found: list[tuple[tuple[int, int], str, Path]] = []
    for path in checkpoint_dir.glob(
        "checkpoint_epoch*_step*.feature_extractor.pth"
    ):
        match = _CHECKPOINT_RE.match(path.name)
        if match is None:
            continue
        epoch, step = int(match.group(1)), int(match.group(2))
        stage = f"epoch{epoch:03d}_step{step:06d}"
        found.append(((epoch, step), stage, path))
    found.sort(key=lambda item: item[0])
    return found


def _resolve_checkpoint(
    trial_dir: str | Path, checkpoint_stage: str | None
) -> tuple[str, Path]:
    """Pick which feature-extractor checkpoint to evaluate for a trial.

    If ``checkpoint_stage`` is given (e.g. ``"epoch009_step001000"``) it is
    used verbatim; otherwise the checkpoint with the highest (epoch, step) is
    chosen. Returns ``(stage_str, feature_extractor_path)``.
    """
    checkpoint_dir = Path(trial_dir) / "checkpoints"
    if checkpoint_stage is not None:
        path = checkpoint_dir / f"checkpoint_{checkpoint_stage}.feature_extractor.pth"
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint for stage '{checkpoint_stage}' not found at {path}"
            )
        return checkpoint_stage, path
    found = _find_checkpoints(trial_dir)
    if not found:
        raise FileNotFoundError(
            f"No feature_extractor checkpoints found under {checkpoint_dir}"
        )
    _, stage, path = found[-1]
    return stage, path


def _candidate_checkpoints(
    trial_dir: str | Path, checkpoint_stage: str | None
) -> list[dict]:
    """List the checkpoints to evaluate for a trial.

    If ``checkpoint_stage`` is given, only that one stage is returned (no
    within-trial sweep). Otherwise every checkpoint in the trial is returned,
    sorted by (epoch, step), so the caller can score them all and keep the
    best. Each entry is ``{"stage", "epoch", "step", "path"}``.
    """
    if checkpoint_stage is not None:
        _, path = _resolve_checkpoint(trial_dir, checkpoint_stage)
        epoch, step = _parse_stage(checkpoint_stage)
        return [{"stage": checkpoint_stage, "epoch": epoch, "step": step, "path": path}]
    found = _find_checkpoints(trial_dir)
    if not found:
        raise FileNotFoundError(
            f"No feature_extractor checkpoints found under "
            f"{Path(trial_dir) / 'checkpoints'}"
        )
    return [
        {"stage": stage, "epoch": epoch, "step": step, "path": path}
        for (epoch, step), stage, path in found
    ]


def _read_trial_metadata(trial_dir: str | Path) -> dict:
    """Best-effort read of a trial's saved training configs for reporting.

    Reads the YAML directly (no dataclass coupling) so it stays robust to
    schema drift and works for both the contrastive and SimSiam pipelines.
    Every field is optional; a missing file or key yields None.
    """
    import yaml

    configs_dir = Path(trial_dir) / "configs"
    meta = {
        "method": None,
        "train_n_variants": None,
        "train_crop_size": None,
        "info_nce_temperature": None,
        "adam_lr": None,
        # ResNet backbone the encoder was trained with. Must be passed to
        # ResNetFeatureExtractor when loading the checkpoint, or the state_dict
        # won't match (e.g. a resnet34 run has extra BasicBlocks a resnet18
        # module lacks). Defaults to resnet18 when the config is absent, which
        # matches ResNetFeatureExtractor's own default for older trials.
        "backbone": "resnet18",
    }

    def _safe_load(name: str):
        path = configs_dir / name
        if not path.is_file():
            return None
        try:
            with path.open() as f:
                return yaml.safe_load(f)
        except Exception as exc:  # pragma: no cover - reporting only
            logging.warning(f"Could not read {path}: {exc}")
            return None

    data_cfg = _safe_load("data_config.yaml")
    if data_cfg:
        meta["train_n_variants"] = data_cfg.get("atomic_batch_n_variants")
        crop = data_cfg.get("crop_size")
        meta["train_crop_size"] = tuple(crop) if crop is not None else None

    loss_cfg = _safe_load("loss_config.yaml")
    if loss_cfg:
        meta["info_nce_temperature"] = loss_cfg.get("info_nce_temperature")

    optimizer_cfg = _safe_load("optimizer_config.yaml")
    if optimizer_cfg:
        meta["adam_lr"] = optimizer_cfg.get("adam_lr")

    arch_cfg = _safe_load("model_architecture_config.yaml")
    if arch_cfg and arch_cfg.get("backbone") is not None:
        meta["backbone"] = arch_cfg.get("backbone")

    # Heuristic: the contrastive pipeline writes loss_config.yaml (InfoNCE
    # temperature); the SimSiam pipeline does not but still writes the other
    # configs.
    if loss_cfg is not None:
        meta["method"] = "contrastive"
    elif data_cfg is not None or optimizer_cfg is not None:
        meta["method"] = "simsiam"
    return meta


def _resolve_trial_names(
    trial_dirs: list[str], trial_names: list[str] | None
) -> list[str]:
    """Pick a short display/output name per trial, disambiguating collisions."""
    if trial_names is not None:
        return list(trial_names)
    names = [Path(d).name for d in trial_dirs]
    if len(set(names)) == len(names):
        return names
    # Basenames collide (e.g. same trial name under different method dirs):
    # prefix with the parent directory name.
    names = [f"{Path(d).parent.name}_{Path(d).name}" for d in trial_dirs]
    if len(set(names)) != len(names):
        names = [f"{name}_{i}" for i, name in enumerate(names)]
    return names


# Untrained reference encoders, mapped to a ResNetFeatureExtractor weights arg.
_BASELINE_WEIGHTS = {"random": None, "imagenet": "IMAGENET1K_V1"}


def _resolve_baselines(baselines: list[str] | None) -> list[dict]:
    """Validate baseline names and map them to encoder specs.

    Returns ``[{"name", "weights", "stage"}, ...]`` where ``name`` is the
    output-dir/display name, ``weights`` is the ResNetFeatureExtractor argument
    (None = random-init, "IMAGENET1K_V1" = pretrained), and ``stage`` is the
    label shown in the ``best_checkpoint_stage`` column.
    """
    if not baselines:
        return []
    specs = []
    for b in baselines:
        key = b.strip().lower()
        if key not in _BASELINE_WEIGHTS:
            raise ValueError(
                f"Unknown baseline {b!r}; expected one of "
                f"{sorted(_BASELINE_WEIGHTS)}."
            )
        specs.append(
            {
                "name": f"baseline_{key}",
                "weights": _BASELINE_WEIGHTS[key],
                "stage": "random-init" if key == "random" else "imagenet",
            }
        )
    return specs


# --------------------------------------------------------------------------- #
# Core evaluation
# --------------------------------------------------------------------------- #
def _build_eval_loader(
    data_dirs: list[str],
    atomic_batch_n_samples: int,
    atomic_batch_n_variants: int,
    image_size: tuple[int, int],
    batch_size: int,
    n_workers: int | None,
    crop_size: tuple[int, int] | None,
    max_batches: int | None,
):
    """Build the (center-cropped, unshuffled) atomic-batch eval dataloader.

    Returns ``(loader, total)`` where ``total`` is the expected number of
    iterations (for the tqdm bar). The loader is map-style and re-iterable, so
    a single instance can be reused across many checkpoints in a comparison.
    """
    _, loader = init_atomic_dataset_and_dataloader(
        data_dirs=data_dirs,
        atomic_batch_n_samples=atomic_batch_n_samples,
        atomic_batch_n_variants=atomic_batch_n_variants,
        input_image_size=image_size,
        batch_size=batch_size,
        n_workers=n_workers,
        n_channels=3,
        shuffle=False,
        crop_size=tuple(crop_size) if crop_size is not None else None,
        crop_mode="center",
    )
    total = max_batches if max_batches is not None else len(loader)
    return loader, total


def _evaluate_feature_extractor(
    feature_extractor_weights: str | Path | None,
    loader,
    total: int,
    *,
    device: str,
    use_float16: bool,
    max_batches: int | None,
    backbone: str = "resnet18",
) -> dict:
    """Run one feature extractor over the loader and accumulate the metrics.

    Returns a dict with ``invariance_ratio``, ``within_frame_sim``,
    ``between_frame_sim`` (batch-averaged scalars), ``within_matrix`` and
    ``between_matrix`` (batch-averaged V x V tensors), ``feature_spread``, and
    ``n_batches_used``. The invariance metrics are on mean-centered pooled
    features. ``feature_spread`` is the collapse diagnostic: the mean
    per-dimension standard deviation of the L2-normalized (NOT centered) pooled
    features, scaled by sqrt(feature_dim) so it is ~1 for a healthy high-rank
    representation and -> 0 as the representation collapses to a point/subspace.
    (Centering is deliberately skipped here: it would hide a collapse by mapping
    near-constant features to random directions.) Frees the encoder and CUDA
    cache before returning so the next trial starts clean.
    """
    # Pass weights through as-is: a path str/Path loads a checkpoint,
    # "IMAGENET1K_V1" uses the pretrained backbone, None is random-init.
    # ``backbone`` must match the architecture the checkpoint was trained with
    # (read from the trial's model_architecture_config.yaml); a mismatch fails
    # the state_dict load.
    feature_extractor = ResNetFeatureExtractor(
        weights=feature_extractor_weights, backbone=backbone
    )
    feature_extractor.to(device)
    feature_extractor.eval()
    device_type = (
        "cuda" if (torch.cuda.is_available() and "cuda" in str(device)) else "cpu"
    )

    total_invariance_ratio = 0.0
    total_within_sim = 0.0
    total_between_sim = 0.0
    accumulated_within_matrix: torch.Tensor | None = None
    accumulated_between_matrix: torch.Tensor | None = None
    # Running sums for the per-dimension std of L2-normalized features.
    feat_sum: torch.Tensor | None = None
    feat_sq_sum: torch.Tensor | None = None
    n_feature_rows = 0
    n_batches_used = 0

    with torch.no_grad():
        for batch_idx, (atomic_batches, _) in tqdm(
            enumerate(loader), total=total, disable=None
        ):
            if max_batches is not None and batch_idx >= max_batches:
                break
            # crop already applied per atomic batch inside the workers
            atomic_batches = atomic_batches.to(device, non_blocking=True)
            concatenated_batch = concat_atomic_batches(atomic_batches)
            n_variants, n_samples, _, _, _ = concatenated_batch.shape
            collapsed_batch = collapse_batch(concatenated_batch)

            with torch.amp.autocast(device_type, enabled=use_float16):
                h = feature_extractor(collapsed_batch)
                h_pooled = F.adaptive_avg_pool2d(h, (1, 1)).flatten(start_dim=1)

            h_pooled = h_pooled.float()
            metrics = compute_alignment_metrics(
                h_pooled,
                n_samples=n_samples,
                n_variants=n_variants,
            )
            # Collapse diagnostic accumulators, on raw L2-normalized features.
            h_norm = F.normalize(h_pooled, dim=1)
            if feat_sum is None:
                feat_sum = h_norm.sum(dim=0)
                feat_sq_sum = (h_norm**2).sum(dim=0)
            else:
                feat_sum = feat_sum + h_norm.sum(dim=0)
                feat_sq_sum = feat_sq_sum + (h_norm**2).sum(dim=0)
            n_feature_rows += h_norm.shape[0]
            total_invariance_ratio += float(metrics.invariance_ratio)
            total_within_sim += float(metrics.within_frame_sim)
            total_between_sim += float(metrics.between_frame_sim)
            if accumulated_within_matrix is None:
                accumulated_within_matrix = metrics.within_frame_matrix
                accumulated_between_matrix = metrics.between_frame_matrix
            else:
                accumulated_within_matrix = (
                    accumulated_within_matrix + metrics.within_frame_matrix
                )
                accumulated_between_matrix = (
                    accumulated_between_matrix + metrics.between_frame_matrix
                )
            n_batches_used += 1

    if n_batches_used == 0:
        raise RuntimeError(
            "No batches were consumed. Check data_dirs and max_batches."
        )

    del feature_extractor
    if device_type == "cuda":
        torch.cuda.empty_cache()

    # feature_spread = mean per-dim std of L2-normalized features * sqrt(d):
    # ~1 for a healthy high-rank representation, -> 0 under collapse.
    feat_mean = feat_sum / n_feature_rows
    feat_var = (feat_sq_sum / n_feature_rows - feat_mean**2).clamp_min(0.0)
    feature_dim = feat_mean.numel()
    feature_spread = float(feat_var.sqrt().mean() * (feature_dim**0.5))

    return {
        "invariance_ratio": total_invariance_ratio / n_batches_used,
        "within_frame_sim": total_within_sim / n_batches_used,
        "between_frame_sim": total_between_sim / n_batches_used,
        "within_matrix": accumulated_within_matrix / n_batches_used,
        "between_matrix": accumulated_between_matrix / n_batches_used,
        "feature_spread": feature_spread,
        "n_batches_used": n_batches_used,
    }


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def _write_checkpoint_outputs(
    output_dir: str | Path,
    *,
    feature_extractor_weights: str,
    data_dirs: list[str],
    atomic_batch_n_variants: int,
    atomic_batch_n_samples: int,
    image_size: tuple[int, int],
    crop_size: tuple[int, int] | None,
    avg_ratio: float,
    within_frame_sim: float,
    between_frame_sim: float,
    within_matrix: torch.Tensor,
    between_matrix: torch.Tensor,
    feature_spread: float,
    n_batches_used: int,
) -> tuple[Path, Path, Path]:
    """Write the within/between heatmap PNGs and the text summary.

    Returns ``(within_heatmap_path, between_heatmap_path, summary_path)``.
    """
    from PIL import Image

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    within_heatmap_path = output_dir_path / "within_frame_similarity_heatmap.png"
    Image.fromarray(
        _render_variant_similarity_heatmap(
            within_matrix, title="Within-frame cosine similarity"
        )
    ).save(within_heatmap_path)
    between_heatmap_path = output_dir_path / "between_frame_similarity_heatmap.png"
    Image.fromarray(
        _render_variant_similarity_heatmap(
            between_matrix, title="Between-frame cosine similarity"
        )
    ).save(between_heatmap_path)

    summary_path = output_dir_path / "variant_similarity_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"feature_extractor_weights: {feature_extractor_weights}\n")
        f.write("data_dirs:\n")
        for d in data_dirs:
            f.write(f"  - {d}\n")
        f.write(f"n_batches_used: {n_batches_used}\n")
        f.write(f"atomic_batch_n_variants: {atomic_batch_n_variants}\n")
        f.write(f"atomic_batch_n_samples: {atomic_batch_n_samples}\n")
        f.write(f"image_size: {tuple(image_size)}\n")
        f.write(
            f"crop_size: {tuple(crop_size) if crop_size is not None else None}\n"
        )
        f.write("(metrics on mean-centered pooled features)\n")
        f.write(f"invariance_ratio (lower = more invariant): {avg_ratio:.6f}\n")
        f.write(f"within_frame_sim (higher = variants align): {within_frame_sim:.6f}\n")
        f.write(f"between_frame_sim (the floor): {between_frame_sim:.6f}\n")
        f.write(
            f"feature_spread (~1 healthy, ->0 collapsed): {feature_spread:.6f}\n"
        )
        for name, matrix in (
            ("within_frame_matrix", within_matrix),
            ("between_frame_matrix", between_matrix),
        ):
            f.write(f"{name}:\n")
            m = matrix.detach().cpu().to(torch.float32).numpy()
            for i in range(m.shape[0]):
                row = " ".join(f"{m[i, j]:.4f}" for j in range(m.shape[1]))
                f.write(f"  v{i}: {row}\n")

    return within_heatmap_path, between_heatmap_path, summary_path


def _write_trial_checkpoint_scores(
    output_dir: str | Path, per_checkpoint: list[dict], best_stage: str
) -> Path:
    """Write the within-trial invariance-ratio trajectory over checkpoints.

    ``per_checkpoint`` entries are the merged candidate + evaluation dicts
    (``stage``, ``epoch``, ``step``, ``invariance_ratio``, ``within_frame_sim``,
    ``between_frame_sim``, ``n_batches_used``). The row whose stage equals
    ``best_stage`` is flagged ``is_best=True``.
    """
    import csv

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir_path / "checkpoint_scores.csv"

    ordered = sorted(
        per_checkpoint,
        key=lambda c: (
            c.get("epoch") if c.get("epoch") is not None else -1,
            c.get("step") if c.get("step") is not None else -1,
        ),
    )
    with scores_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stage",
                "epoch",
                "step",
                "invariance_ratio",
                "within_frame_sim",
                "between_frame_sim",
                "feature_spread",
                "n_batches_used",
                "is_best",
            ]
        )
        for entry in ordered:
            writer.writerow(
                [
                    entry["stage"],
                    entry.get("epoch"),
                    entry.get("step"),
                    entry["invariance_ratio"],
                    entry["within_frame_sim"],
                    entry["between_frame_sim"],
                    entry["feature_spread"],
                    entry["n_batches_used"],
                    entry["stage"] == best_stage,
                ]
            )
    return scores_path


def _render_comparison_barchart(names: list[str], ratios: list[float]):
    """Render a horizontal bar chart of invariance ratio per trial.

    Trials are sorted best-first (lowest ratio on top). Returns an HWC uint8
    RGB array.
    """
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    order = sorted(range(len(names)), key=lambda i: ratios[i])
    sorted_names = [names[i] for i in order]
    sorted_ratios = [ratios[i] for i in order]
    n = len(sorted_names)

    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.5 * max(n, 1)), dpi=100)
    y = list(range(n))
    ax.barh(y, sorted_ratios, color="#4c72b0")
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()  # best (lowest) at the top
    ax.set_xlabel("Invariance ratio (lower = more style-invariant)")
    ax.set_title("Variant-similarity comparison across trials")
    for yi, r in zip(y, sorted_ratios):
        ax.text(r, yi, f" {r:.4f}", va="center", ha="left", fontsize=9)
    ax.margins(x=0.18)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def _format_cell(value) -> str:
    """Format a metadata/metric value for the aligned text table."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_comparison_outputs(
    output_dir: str | Path, rows: list[dict], eval_spec: dict
) -> tuple[Path, Path, Path | None]:
    """Write the comparison CSV, ranked text summary, and bar chart.

    ``rows`` includes failed trials (``invariance_ratio is None``); they appear
    in the CSV and summary but are excluded from the ranking and chart.
    """
    import csv

    from PIL import Image

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "trial_name",
        "method",
        "best_checkpoint_stage",
        "invariance_ratio",
        "selection_invariance_ratio",
        "within_frame_sim",
        "between_frame_sim",
        "feature_spread",
        "n_checkpoints_evaluated",
        "selection",
        "n_batches_used",
        "train_n_variants",
        "train_crop_size",
        "info_nce_temperature",
        "adam_lr",
        "status",
        "trial_dir",
    ]
    csv_path = output_dir_path / "comparison.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    ok_rows = [r for r in rows if r.get("invariance_ratio") is not None]
    chart_path: Path | None = None
    if ok_rows:
        chart = _render_comparison_barchart(
            [r["trial_name"] for r in ok_rows],
            [r["invariance_ratio"] for r in ok_rows],
        )
        chart_path = output_dir_path / "comparison_invariance_ratio.png"
        Image.fromarray(chart).save(chart_path)

    summary_path = output_dir_path / "comparison_summary.txt"
    table_columns = [
        ("rank", "rank"),
        ("trial", "trial_name"),
        ("method", "method"),
        ("best_ckpt", "best_checkpoint_stage"),
        ("inv_ratio", "invariance_ratio"),
        ("sel_ratio", "selection_invariance_ratio"),
        ("within_sim", "within_frame_sim"),
        ("between_sim", "between_frame_sim"),
        ("feat_spread", "feature_spread"),
        ("n_ckpts", "n_checkpoints_evaluated"),
        ("train_nvar", "train_n_variants"),
        ("train_crop", "train_crop_size"),
        ("infonce_T", "info_nce_temperature"),
        ("adam_lr", "adam_lr"),
    ]
    sorted_ok = sorted(ok_rows, key=lambda r: r["invariance_ratio"])
    failed_rows = [r for r in rows if r.get("invariance_ratio") is None]
    display_rows = sorted_ok + failed_rows

    # Build cell strings, then size each column to its widest entry.
    header_cells = [header for header, _ in table_columns]
    body = [
        [_format_cell(row.get(key)) for _, key in table_columns]
        for row in display_rows
    ]
    widths = [
        max(len(header_cells[i]), *(len(r[i]) for r in body)) if body else len(header_cells[i])
        for i in range(len(table_columns))
    ]

    def _fmt_line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    with summary_path.open("w") as f:
        f.write("Variant-similarity comparison across contrastive learning trials\n")
        f.write("Lower invariance ratio = more style-invariant features (better).\n")
        f.write(
            "Ranked by inv_ratio = the invariance ratio on the COMPARISON set.\n"
            "sel_ratio = the chosen checkpoint's ratio on the SELECTION set (where it\n"
            "was picked); a much lower sel_ratio than inv_ratio means the selection\n"
            "overfit that set. method='baseline' rows are untrained encoders (random /\n"
            "ImageNet) — every trained trial should rank below (beat) them.\n"
            "feat_spread = collapse check (~1 healthy, ->0 collapsed); a low\n"
            "inv_ratio paired with a low feat_spread is a collapse artifact, not\n"
            "a good representation.\n\n"
        )
        f.write("Shared evaluation spec (identical for every trial):\n")
        for key, value in eval_spec.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        f.write(_fmt_line(header_cells) + "\n")
        f.write("-" * (sum(widths) + 2 * (len(widths) - 1)) + "\n")
        for cells in body:
            f.write(_fmt_line(cells) + "\n")
        if failed_rows:
            f.write("\nFailed trials:\n")
            for row in failed_rows:
                f.write(f"  {row['trial_name']}: {row.get('status')}\n")

    return csv_path, summary_path, chart_path


# --------------------------------------------------------------------------- #
# Public entrypoints
# --------------------------------------------------------------------------- #
def compute_variant_similarity(
    feature_extractor_weights: str,
    data_dirs: list[str],
    atomic_batch_n_samples: int,
    atomic_batch_n_variants: int,
    batch_size: int,
    image_size: tuple[int, int],
    output_dir: str,
    crop_size: tuple[int, int] | None = None,
    crop_border_exclude: int = 0,
    max_batches: int | None = None,
    n_workers: int | None = 4,
    device: str = "cuda",
    use_float16: bool = True,
    backbone: str = "resnet18",
) -> None:
    """Compute the within/between variant-similarity heatmaps and invariance
    ratio for a single checkpoint.

    Args:
        feature_extractor_weights: Path to a feature_extractor.pth saved
            by either the contrastive or the SimSiam pipeline.
        data_dirs: Directories of atomic batches to use as the evaluation
            set. Typically the validation set used during training.
        atomic_batch_n_samples: Frames per atomic batch (must match what
            was used when the atomic batches were extracted).
        atomic_batch_n_variants: Style variants per frame.
        batch_size: Number of atomic batches to read per step.
        image_size: (H, W) the frames are loaded at.
        output_dir: Where to write the two heatmap PNGs and a text summary.
        crop_size: If set, apply an aligned center-crop of this size to
            every frame before encoding. Should match the crop the
            training run used.
        crop_border_exclude: Unused for center cropping but kept in the
            CLI for parity with the training config.
        max_batches: Cap on how many batches to consume; None = all.
        n_workers: Dataloader workers.
        device: "cuda" or "cpu".
        use_float16: Run the encoder in mixed precision.
        backbone: ResNet variant the checkpoint was trained with, one of
            "resnet18" (default) or "resnet34". Must match, or loading the
            state_dict fails. ``compare`` reads this automatically from each
            trial's config; here it is explicit since only a raw weights path
            is given.
    """
    del crop_border_exclude  # parity with the training config; not used here

    logging.info(f"Building dataloader over {len(data_dirs)} dirs")
    loader, total = _build_eval_loader(
        data_dirs,
        atomic_batch_n_samples,
        atomic_batch_n_variants,
        image_size,
        batch_size,
        n_workers,
        crop_size,
        max_batches,
    )

    logging.info(f"Loading feature extractor from {feature_extractor_weights}")
    result = _evaluate_feature_extractor(
        feature_extractor_weights,
        loader,
        total,
        device=device,
        use_float16=use_float16,
        max_batches=max_batches,
        backbone=backbone,
    )

    within_heatmap_path, between_heatmap_path, summary_path = _write_checkpoint_outputs(
        output_dir,
        feature_extractor_weights=str(feature_extractor_weights),
        data_dirs=data_dirs,
        atomic_batch_n_variants=atomic_batch_n_variants,
        atomic_batch_n_samples=atomic_batch_n_samples,
        image_size=image_size,
        crop_size=crop_size,
        avg_ratio=result["invariance_ratio"],
        within_frame_sim=result["within_frame_sim"],
        between_frame_sim=result["between_frame_sim"],
        within_matrix=result["within_matrix"],
        between_matrix=result["between_matrix"],
        feature_spread=result["feature_spread"],
        n_batches_used=result["n_batches_used"],
    )

    logging.info(
        f"Invariance ratio: {result['invariance_ratio']:.4f} (lower is better); "
        f"within_frame_sim {result['within_frame_sim']:.4f}, "
        f"between_frame_sim {result['between_frame_sim']:.4f}; "
        f"feature_spread {result['feature_spread']:.4f} (~1 healthy, ->0 collapsed); "
        f"heatmaps saved to {within_heatmap_path} and {between_heatmap_path}; "
        f"summary to {summary_path}"
    )


def compare_trials(
    trial_dirs: list[str],
    data_dirs: list[str],
    atomic_batch_n_samples: int,
    atomic_batch_n_variants: int,
    batch_size: int,
    image_size: tuple[int, int],
    output_dir: str,
    crop_size: tuple[int, int] | None = None,
    crop_border_exclude: int = 0,
    checkpoint_stage: str | None = None,
    selection_data_dirs: list[str] | None = None,
    baselines: list[str] | None = None,
    trial_names: list[str] | None = None,
    max_batches: int | None = None,
    selection_max_batches: int | None = None,
    min_feature_spread: float | None = None,
    n_workers: int | None = 4,
    device: str = "cuda",
    use_float16: bool = True,
) -> None:
    """Benchmark many training trials head-to-head and rank them.

    Two-stage, with an optional held-out split that removes selection bias:

    1. SELECT. Within each trial, every checkpoint is scored on the *selection*
       set (``selection_data_dirs`` if given, else ``data_dirs``) and the
       best-scoring one (lowest invariance ratio) is kept as the trial's
       representative. This avoids penalising a run whose final checkpoint is
       past its peak. Pin a single stage with ``checkpoint_stage`` to skip the
       sweep.
    2. SCORE & RANK. Each trial's chosen checkpoint is scored on the
       *comparison* set (``data_dirs``) and ranked. When a separate
       ``selection_data_dirs`` is given, the checkpoint is picked on one fly and
       scored on another, so the reported number is not the cherry-picked
       maximum on the same data — important when trials have different numbers
       of checkpoints (more checkpoints = more chances at a lucky low).

    Optionally, untrained ``baselines`` (random-init and/or ImageNet ResNet) are
    scored on the comparison set and ranked alongside the trials, so you can see
    whether the trained encoders beat a trivial one at all.

    Only one dataloader is held at a time (selection, then comparison) to keep
    memory bounded; each is unshuffled so every evaluation sees identical inputs.

    Per trial, ``output_dir/<trial_name>/`` gets the chosen checkpoint's
    comparison-set heatmaps + summary and a ``checkpoint_scores.csv`` trajectory
    of the selection-set sweep. Baselines get ``output_dir/baseline_<name>/``. A
    ranked ``comparison.csv``, ``comparison_summary.txt``, and
    ``comparison_invariance_ratio.png`` are written at the top of ``output_dir``.
    A trial that fails (no checkpoint, bad config, OOM, ...) is logged and
    recorded as failed but does not abort the rest of the comparison.

    Args:
        trial_dirs: Trial output directories, each containing ``checkpoints/``
            and (optionally) ``configs/``. Works for both contrastive and
            SimSiam trials.
        data_dirs: The *comparison* set — atomic-batch dirs every finalist is
            scored and ranked on. Identical for every trial/baseline.
        atomic_batch_n_samples: Frames per atomic batch (must match extraction).
        atomic_batch_n_variants: Style variants per frame in the eval data.
        batch_size: Number of atomic batches to read per step.
        image_size: (H, W) the frames are loaded at.
        output_dir: Where to write the comparison table, chart, and per-trial
            artifacts.
        crop_size: Shared aligned center-crop applied to every frame before
            encoding (None = full frame). Use the same value for all trials.
        crop_border_exclude: Unused for center cropping; kept for CLI parity.
        checkpoint_stage: If set (e.g. ``"epoch009_step001000"``), evaluate only
            this exact stage in every trial (no within-trial sweep); otherwise
            sweep all checkpoints per trial and keep each trial's best.
        selection_data_dirs: Optional *selection* set — a held-out set, distinct
            from ``data_dirs``, used only to pick each trial's best checkpoint.
            Strongly recommended for an unbiased ranking; if None, selection and
            comparison both use ``data_dirs`` (cherry-picks on the test set).
        baselines: Untrained encoders to rank alongside the trials, any of
            ``"random"`` (random-init ResNet) and ``"imagenet"`` (ImageNet
            ResNet). None = no baselines.
        trial_names: Optional display/output names parallel to ``trial_dirs``;
            defaults to the directory basenames (disambiguated if they collide).
        max_batches: Cap on how many batches to consume in the *comparison*
            eval (the headline score); None = all.
        selection_max_batches: Cap on how many batches to consume per checkpoint
            during the *selection* sweep; None = use ``max_batches``. The sweep
            is the dominant cost (every checkpoint of every trial), and it only
            needs to *rank* checkpoints, so a smaller cap here (e.g. 20-30) cuts
            runtime a lot. Whenever it differs from ``max_batches``, each trial's
            chosen checkpoint is re-scored on the comparison set at
            ``max_batches`` for the headline number — so this works on a single
            eval set too (cheap sweep, thorough final score), not only with a
            separate ``selection_data_dirs``.
        min_feature_spread: Collapse guard for selection. ``feature_spread`` is
            ~1 for a healthy high-rank representation and -> 0 as it collapses.
            If set (e.g. 0.2), checkpoints whose selection-set spread is below
            this are excluded before picking the lowest invariance ratio — a
            collapsed encoder can post a deceptively low ratio. If every
            checkpoint is below it, the trial is kept (best ratio) with a
            warning. None = no filtering (still reported, and a hard-collapse
            warning fires if the chosen checkpoint's spread is < 0.1). Calibrate
            against the ``random``/``imagenet`` baselines' reported spread.
        n_workers: Dataloader workers.
        device: "cuda" or "cpu".
        use_float16: Run the encoder in mixed precision.
    """
    del crop_border_exclude  # parity with the training config; not used here

    if trial_names is not None and len(trial_names) != len(trial_dirs):
        raise ValueError(
            f"trial_names ({len(trial_names)}) must match trial_dirs "
            f"({len(trial_dirs)}) in length"
        )
    baseline_specs = _resolve_baselines(baselines)
    if len(trial_dirs) == 0 and not baseline_specs:
        raise ValueError("Nothing to compare: no trial_dirs and no baselines.")

    names = _resolve_trial_names(trial_dirs, trial_names)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    use_separate_selection = selection_data_dirs is not None
    selection_dirs = selection_data_dirs if use_separate_selection else data_dirs
    # The per-checkpoint selection sweep may use a cheaper cap than the final
    # comparison eval: the sweep only needs to RANK a trial's checkpoints, while
    # the chosen one is then scored thoroughly. This holds whether or not the
    # selection set is a separate held-out video.
    sel_cap = (
        selection_max_batches if selection_max_batches is not None else max_batches
    )
    # The selection eval can be reused as the comparison score only when it
    # genuinely IS that eval: same data AND same cap. Otherwise the chosen
    # checkpoint is re-scored on the comparison loader at `max_batches`.
    reuse_selection = (not use_separate_selection) and (sel_cap == max_batches)

    def _blank_row(**overrides) -> dict:
        row = {
            "trial_name": None,
            "trial_dir": None,
            "method": None,
            "train_n_variants": None,
            "train_crop_size": None,
            "info_nce_temperature": None,
            "adam_lr": None,
            "best_checkpoint_stage": None,
            "invariance_ratio": None,
            "selection_invariance_ratio": None,
            "within_frame_sim": None,
            "between_frame_sim": None,
            "feature_spread": None,
            "n_checkpoints_evaluated": None,
            "selection": None,
            "n_batches_used": None,
            "status": "ok",
        }
        row.update(overrides)
        return row

    rows: list[dict] = []
    # Each finalist: {row, name, weights, weights_label, is_baseline, comp_metrics}.
    # comp_metrics is pre-filled only when selection == comparison (no re-eval).
    finalists: list[dict] = []

    # ---------------------------------------------------------------- #
    # Phase 1: pick each trial's best checkpoint on the SELECTION set.
    # ---------------------------------------------------------------- #
    logging.info(
        f"[selection] building dataloader over {len(selection_dirs)} dir(s)"
        + (" (held-out selection set)" if use_separate_selection else "")
    )
    sel_loader, sel_total = _build_eval_loader(
        selection_dirs,
        atomic_batch_n_samples,
        atomic_batch_n_variants,
        image_size,
        batch_size,
        n_workers,
        crop_size,
        sel_cap,
    )
    for trial_dir, name in zip(trial_dirs, names):
        logging.info(f"=== Selecting checkpoint for trial '{name}' ({trial_dir}) ===")
        meta = _read_trial_metadata(trial_dir)
        row = _blank_row(
            trial_name=name,
            trial_dir=trial_dir,
            method=meta["method"],
            train_n_variants=meta["train_n_variants"],
            train_crop_size=meta["train_crop_size"],
            info_nce_temperature=meta["info_nce_temperature"],
            adam_lr=meta["adam_lr"],
            selection="pinned" if checkpoint_stage is not None else "best_of_n",
        )
        try:
            candidates = _candidate_checkpoints(trial_dir, checkpoint_stage)
            logging.info(
                f"Trial '{name}': scoring {len(candidates)} checkpoint(s) "
                "on the selection set"
            )
            per_checkpoint: list[dict] = []
            for candidate in candidates:
                sel = _evaluate_feature_extractor(
                    candidate["path"],
                    sel_loader,
                    sel_total,
                    device=device,
                    use_float16=use_float16,
                    max_batches=sel_cap,
                    backbone=meta["backbone"],
                )
                per_checkpoint.append({**candidate, **sel})
                logging.info(
                    f"Trial '{name}' [{candidate['stage']}]: selection "
                    f"invariance ratio {sel['invariance_ratio']:.4f}, "
                    f"feature_spread {sel['feature_spread']:.4f}"
                )

            # Collapse guard: a collapsed encoder can post a deceptively low
            # invariance ratio, so drop low-spread checkpoints before picking the
            # lowest ratio. If that empties the pool, keep all (and warn).
            candidates_pool = per_checkpoint
            if min_feature_spread is not None:
                healthy = [
                    c
                    for c in per_checkpoint
                    if c["feature_spread"] >= min_feature_spread
                ]
                n_dropped = len(per_checkpoint) - len(healthy)
                if n_dropped:
                    logging.info(
                        f"Trial '{name}': collapse guard dropped {n_dropped}/"
                        f"{len(per_checkpoint)} checkpoint(s) with feature_spread "
                        f"< {min_feature_spread}"
                    )
                if healthy:
                    candidates_pool = healthy
                else:
                    logging.warning(
                        f"Trial '{name}': ALL checkpoints have feature_spread "
                        f"< {min_feature_spread} — representation may have "
                        "collapsed. Selecting best ratio among all anyway."
                    )

            # Keep the best-scoring checkpoint (on the selection set).
            best = min(candidates_pool, key=lambda c: c["invariance_ratio"])
            if best["feature_spread"] < 0.1:
                logging.warning(
                    f"Trial '{name}': selected {best['stage']} has very low "
                    f"feature_spread ({best['feature_spread']:.4f}) — likely "
                    "near-collapse; its low invariance ratio may be an artifact."
                )
            _write_trial_checkpoint_scores(
                output_dir_path / name, per_checkpoint, best_stage=best["stage"]
            )
            row["best_checkpoint_stage"] = best["stage"]
            row["selection_invariance_ratio"] = best["invariance_ratio"]
            row["n_checkpoints_evaluated"] = len(per_checkpoint)
            finalists.append(
                {
                    "row": row,
                    "name": name,
                    "weights": best["path"],
                    "weights_label": str(best["path"]),
                    "backbone": meta["backbone"],
                    "is_baseline": False,
                    # Reuse this eval as the comparison score only when it IS the
                    # comparison eval (same data + same cap); else re-score later.
                    "comp_metrics": best if reuse_selection else None,
                }
            )
            logging.info(
                f"Trial '{name}': selected {best['stage']} (selection ratio "
                f"{best['invariance_ratio']:.4f} of {len(per_checkpoint)})"
            )
        except Exception as exc:
            logging.exception(f"Trial '{name}' selection failed: {exc}")
            row["status"] = f"failed: {type(exc).__name__}: {exc}"
        rows.append(row)

    # Switch to the comparison loader. When the selection set is separate, free
    # its workers first so only one full-resolution loader is resident at a time.
    if use_separate_selection:
        del sel_loader
        logging.info(f"[comparison] building dataloader over {len(data_dirs)} dir(s)")
        comp_loader, comp_total = _build_eval_loader(
            data_dirs,
            atomic_batch_n_samples,
            atomic_batch_n_variants,
            image_size,
            batch_size,
            n_workers,
            crop_size,
            max_batches,
        )
    else:
        # Same data as selection: reuse the loader object. The comparison eval
        # may use a larger cap than the selection sweep, so set the tqdm total
        # to the comparison cap (the loader itself is un-truncated/re-iterable).
        comp_loader = sel_loader
        comp_total = (
            sel_total
            if reuse_selection
            else (max_batches if max_batches is not None else len(sel_loader))
        )

    # Queue untrained baselines as finalists scored on the comparison set.
    for spec in baseline_specs:
        row = _blank_row(
            trial_name=spec["name"],
            trial_dir="(baseline)",
            method="baseline",
            best_checkpoint_stage=spec["stage"],
            n_checkpoints_evaluated=1,
            selection="baseline",
        )
        rows.append(row)
        finalists.append(
            {
                "row": row,
                "name": spec["name"],
                "weights": spec["weights"],
                "weights_label": f"{spec['stage']} (untrained baseline)",
                # Baselines load no checkpoint state_dict (random-init or
                # torchvision ImageNet weights), so the backbone is just the
                # reference architecture; resnet18 matches the historic default.
                "backbone": "resnet18",
                "is_baseline": True,
                "comp_metrics": None,
            }
        )

    # ---------------------------------------------------------------- #
    # Phase 2: score every finalist on the COMPARISON set and rank.
    # ---------------------------------------------------------------- #
    for fin in finalists:
        row = fin["row"]
        name = fin["name"]
        try:
            comp = fin["comp_metrics"]
            if comp is None:
                logging.info(f"[comparison] scoring '{name}'")
                comp = _evaluate_feature_extractor(
                    fin["weights"],
                    comp_loader,
                    comp_total,
                    device=device,
                    use_float16=use_float16,
                    max_batches=max_batches,
                    backbone=fin["backbone"],
                )
            row["invariance_ratio"] = comp["invariance_ratio"]
            row["within_frame_sim"] = comp["within_frame_sim"]
            row["between_frame_sim"] = comp["between_frame_sim"]
            row["feature_spread"] = comp["feature_spread"]
            row["n_batches_used"] = comp["n_batches_used"]
            if fin["is_baseline"]:
                # Baselines have no separate selection step.
                row["selection_invariance_ratio"] = comp["invariance_ratio"]
            _write_checkpoint_outputs(
                output_dir_path / name,
                feature_extractor_weights=fin["weights_label"],
                data_dirs=data_dirs,
                atomic_batch_n_variants=atomic_batch_n_variants,
                atomic_batch_n_samples=atomic_batch_n_samples,
                image_size=image_size,
                crop_size=crop_size,
                avg_ratio=comp["invariance_ratio"],
                within_frame_sim=comp["within_frame_sim"],
                between_frame_sim=comp["between_frame_sim"],
                within_matrix=comp["within_matrix"],
                between_matrix=comp["between_matrix"],
                feature_spread=comp["feature_spread"],
                n_batches_used=comp["n_batches_used"],
            )
            logging.info(
                f"'{name}': comparison invariance ratio "
                f"{comp['invariance_ratio']:.4f}, feature_spread "
                f"{comp['feature_spread']:.4f}"
            )
        except Exception as exc:
            logging.exception(f"'{name}' comparison scoring failed: {exc}")
            row["status"] = f"failed: {type(exc).__name__}: {exc}"

    # Rank successful trials best-first (lowest invariance ratio).
    ok_rows = sorted(
        (r for r in rows if r["invariance_ratio"] is not None),
        key=lambda r: r["invariance_ratio"],
    )
    for rank, row in enumerate(ok_rows, start=1):
        row["rank"] = rank

    eval_spec = {
        "comparison_data_dirs": list(data_dirs),
        "selection_data_dirs": (
            list(selection_data_dirs)
            if use_separate_selection
            else "(same as comparison set — selection NOT held out)"
        ),
        "baselines": [s["name"] for s in baseline_specs] or "none",
        "atomic_batch_n_samples": atomic_batch_n_samples,
        "atomic_batch_n_variants": atomic_batch_n_variants,
        "batch_size": batch_size,
        "image_size": tuple(image_size),
        "crop_size": tuple(crop_size) if crop_size is not None else None,
        "max_batches": max_batches,
        "checkpoint_selection": (
            f"pinned stage '{checkpoint_stage}'"
            if checkpoint_stage is not None
            else "best checkpoint per trial (lowest invariance ratio on selection set)"
        ),
        "collapse_guard": (
            f"exclude checkpoints with feature_spread < {min_feature_spread}"
            if min_feature_spread is not None
            else "off (feature_spread reported; warn if chosen < 0.1)"
        ),
    }
    csv_path, summary_path, chart_path = _write_comparison_outputs(
        output_dir, rows, eval_spec
    )

    n_ok = len(ok_rows)
    n_failed = len(rows) - n_ok
    logging.info(
        f"Compared {len(rows)} entries ({n_ok} ok, {n_failed} failed). "
        f"Table: {csv_path}; summary: {summary_path}"
        + (f"; chart: {chart_path}" if chart_path is not None else "")
    )
    if ok_rows:
        best = ok_rows[0]
        logging.info(
            f"Best (lowest invariance ratio): '{best['trial_name']}' "
            f"= {best['invariance_ratio']:.4f}"
        )
        # Warn loudly if no trained trial beat an untrained baseline.
        best_trained = next(
            (r for r in ok_rows if r["method"] != "baseline"), None
        )
        top_baseline = next(
            (r for r in ok_rows if r["method"] == "baseline"), None
        )
        if (
            best_trained is not None
            and top_baseline is not None
            and top_baseline["invariance_ratio"] <= best_trained["invariance_ratio"]
        ):
            logging.warning(
                "No trained trial beat the untrained baseline "
                f"'{top_baseline['trial_name']}' "
                f"({top_baseline['invariance_ratio']:.4f}) — the pretraining may "
                "not be learning style-invariance on this eval."
            )


if __name__ == "__main__":
    from tyro.extras import subcommand_cli_from_dict

    subcommand_cli_from_dict(
        {
            "single": compute_variant_similarity,
            "compare": compare_trials,
        },
        prog=f"python {Path(__file__).name}",
        description=(
            "Compute the within/between variant-similarity heatmaps and "
            "aggregate invariance ratio for one checkpoint ('single'), or "
            "benchmark many training trials head-to-head ('compare')."
        ),
    )

    # Example usage (single checkpoint):
    # python -u src/poseforge/pose/contrast/scripts/compute_variant_similarity.py single \
    #     --feature-extractor-weights "/scratch/.../checkpoint_epoch009_step001000.feature_extractor.pth" \
    #     --data-dirs \
    #         "/work/.../atomic_batches/BO_Gal4_fly1_trial002" \
    #     --atomic-batch-n-samples 32 \
    #     --atomic-batch-n-variants 4 \
    #     --batch-size 32 \
    #     --image-size 900 900 \
    #     --crop-size 256 256 \
    #     --output-dir "/scratch/.../variant_similarity_eval"
    #
    # Example usage (compare many trials):
    # python -u src/poseforge/pose/contrast/scripts/compute_variant_similarity.py compare \
    #     --trial-dirs \
    #         "/scratch/.../contrastive_pretraining/trial_20260622_cropped256" \
    #         "/scratch/.../simsiam_pretraining/trial_20260622_simsiam256" \
    #     --data-dirs \
    #         "/work/.../atomic_batches/BO_Gal4_fly1_trial002" \
    #     --atomic-batch-n-samples 32 \
    #     --atomic-batch-n-variants 2 \
    #     --batch-size 128 \
    #     --image-size 900 900 \
    #     --crop-size 256 256 \
    #     --output-dir "/scratch/.../variant_similarity_comparison/20260624"
