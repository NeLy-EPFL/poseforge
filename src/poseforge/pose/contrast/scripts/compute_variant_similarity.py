"""Compute the variant-similarity heatmap and the aggregate invariance
ratio for a trained feature extractor checkpoint.

Method-agnostic: works for any checkpoint saved by the contrastive
pipeline or the SimSiam pipeline (both write `feature_extractor.pth` in the
same format). The script:

1. Builds a ResNetFeatureExtractor and loads the given weights.
2. Iterates atomic batches in the given data dirs, applies the same
   aligned center-crop logic used at validation time.
3. Pools features the same way the training pipeline does.
4. Calls the shared compute_alignment_metrics helper and accumulates the
   per-variant similarity matrix across batches.
5. Renders the heatmap to a PNG and prints the aggregate invariance ratio.

Use this to:
- inspect the per-variant similarity matrix of a finished training run
  outside of TensorBoard, and to pick variants that should be dropped
  from future runs.
- benchmark two training methods (e.g. contrastive vs SimSiam) on the same
  held-out data with the same code path so the comparison is fair.
"""

import logging
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
) -> None:
    """Compute the per-variant similarity heatmap and invariance ratio.

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
        output_dir: Where to write the heatmap PNG and a text summary.
        crop_size: If set, apply an aligned center-crop of this size to
            every frame before encoding. Should match the crop the
            training run used.
        crop_border_exclude: Unused for center cropping but kept in the
            CLI for parity with the training config.
        max_batches: Cap on how many batches to consume; None = all.
        n_workers: Dataloader workers.
        device: "cuda" or "cpu".
        use_float16: Run the encoder in mixed precision.
    """
    del crop_border_exclude  # parity with the training config; not used here

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading feature extractor from {feature_extractor_weights}")
    feature_extractor = ResNetFeatureExtractor(weights=feature_extractor_weights)
    feature_extractor.to(device)
    feature_extractor.eval()

    logging.info(f"Building dataloader over {len(data_dirs)} dirs")
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
    device_type = "cuda" if (torch.cuda.is_available() and "cuda" in str(device)) else "cpu"

    total_invariance_ratio = 0.0
    accumulated_matrix: torch.Tensor | None = None
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

            invariance_ratio, variant_sim_matrix = compute_alignment_metrics(
                h_pooled.float(),
                n_samples=n_samples,
                n_variants=n_variants,
            )
            total_invariance_ratio += float(invariance_ratio)
            if accumulated_matrix is None:
                accumulated_matrix = variant_sim_matrix
            else:
                accumulated_matrix = accumulated_matrix + variant_sim_matrix
            n_batches_used += 1

    if n_batches_used == 0:
        raise RuntimeError("No batches were consumed. Check data_dirs and max_batches.")

    avg_ratio = total_invariance_ratio / n_batches_used
    avg_matrix = accumulated_matrix / n_batches_used

    heatmap_img = _render_variant_similarity_heatmap(avg_matrix)

    # Save outputs.
    from PIL import Image

    heatmap_path = output_dir_path / "variant_similarity_heatmap.png"
    Image.fromarray(heatmap_img).save(heatmap_path)
    summary_path = output_dir_path / "variant_similarity_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"feature_extractor_weights: {feature_extractor_weights}\n")
        f.write(f"data_dirs:\n")
        for d in data_dirs:
            f.write(f"  - {d}\n")
        f.write(f"n_batches_used: {n_batches_used}\n")
        f.write(f"atomic_batch_n_variants: {atomic_batch_n_variants}\n")
        f.write(f"atomic_batch_n_samples: {atomic_batch_n_samples}\n")
        f.write(f"image_size: {tuple(image_size)}\n")
        f.write(
            f"crop_size: {tuple(crop_size) if crop_size is not None else None}\n"
        )
        f.write(f"invariance_ratio (lower = more invariant): {avg_ratio:.6f}\n")
        f.write("variant_similarity_matrix:\n")
        m = avg_matrix.detach().cpu().to(torch.float32).numpy()
        for i in range(m.shape[0]):
            row = " ".join(f"{m[i, j]:.4f}" for j in range(m.shape[1]))
            f.write(f"  v{i}: {row}\n")

    logging.info(
        f"Invariance ratio: {avg_ratio:.4f} (lower is better); "
        f"heatmap saved to {heatmap_path}; summary to {summary_path}"
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(
        compute_variant_similarity,
        prog=f"python {Path(__file__).name}",
        description=(
            "Compute the per-variant similarity heatmap and aggregate invariance "
            "ratio for a trained feature extractor."
        ),
    )

    # Example usage:
    # python -u src/poseforge/pose/contrast/scripts/compute_variant_similarity.py \
    #     --feature-extractor-weights "/scratch/.../checkpoint_epoch009_step001000.feature_extractor.pth" \
    #     --data-dirs \
    #         "/work/.../atomic_batches/BO_Gal4_fly1_trial002" \
    #     --atomic-batch-n-samples 32 \
    #     --atomic-batch-n-variants 4 \
    #     --batch-size 32 \
    #     --image-size 900 900 \
    #     --crop-size 256 256 \
    #     --output-dir "/scratch/.../variant_similarity_eval"
