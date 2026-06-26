# Contrastive Variant-Similarity Comparison

Benchmark many contrastive-learning trials head-to-head and rank them by how
style-invariant their learned features are.

Every trial is evaluated on the **same** held-out atomic batches with the
**same** code path (load `feature_extractor.pth` → pool → `compute_alignment_metrics`),
so the resulting **invariance ratio** is directly comparable across runs. Lower
invariance ratio = more style-invariant features = better. The evaluation is
method-agnostic: contrastive and SimSiam trials can be compared in the same job
because both pipelines write `feature_extractor.pth` in the same format.

The eval runs on the **full frames** (no crop), even though the encoders were
trained on 256×256 crops. The downstream tasks (keypoints3d / bodyseg /
joint_segpose) feed the feature extractor full ~912×912 images with no crop, so
full-frame invariance is the regime we actually deploy in and the fair thing to
rank on. It is a distribution shift from training, but an identical one for
every model, so the comparison stays apples-to-apples. (To instead reproduce
the training-time validation numbers, add `--crop-size 256 256` and raise
`--batch-size`.)

Within each trial, **every checkpoint** is scored and the best-scoring one
(lowest invariance ratio) is kept as that trial's representative — so a run is
judged at its peak, not at whatever its final step happened to be (e.g. a
SimSiam run that began to collapse late in training).

Two safeguards make the ranking trustworthy:

- **Selection ≠ comparison.** The best checkpoint is picked on a *selection*
  video and then re-scored on a *different* held-out *comparison* video, which
  is what the ranking uses. Picking and ranking on the same data cherry-picks
  the luckiest checkpoint and unfairly favours trials that saved more
  checkpoints; splitting the two removes that bias. (Set both to the same set
  to disable.)
- **Baselines.** An untrained random-init ResNet and an ImageNet-pretrained
  ResNet are ranked alongside the trials. Every trained trial should beat both;
  if one doesn't, the pretraining isn't buying style-invariance on this eval.

This wraps the `compare` subcommand of
`src/poseforge/pose/contrast/scripts/compute_variant_similarity.py`.

## Usage

### 1. Configure the trials and the shared eval set

Edit `compare_trials.run`:

- `trial_dirs` — one entry per trained run (a trial output directory containing
  `checkpoints/` and `configs/`). Mix contrastive and SimSiam freely.
- `comparison_data_dirs` (`--data-dirs`) — the held-out video trials are
  **ranked** on. `selection_data_dirs` (`--selection-data-dirs`) — a *different*
  held-out video used only to pick each trial's best checkpoint. Both must be
  held out of every trial's training; full frames; no `--crop-size`.
- `--baselines random imagenet` — untrained reference encoders ranked alongside.
- `--batch-size` is kept small (full frames are ~12× the pixels of a crop);
  lower it further on CUDA OOM.
- `comparison_name` / `output_dir` — where the results are written.
- (optional) add `--checkpoint-stage epochXXX_stepYYYYYY` to pin the same
  checkpoint in every trial (skips the per-trial best-checkpoint sweep).
- (optional) add `--max-batches N` to bound each evaluation — useful for a quick
  smoke test, or to keep the selection sweep cheap when trials have many
  checkpoints.

### 2. Submit

```bash
cd ~/poseforge/scripts_on_cluster/contrastive_variant_similarity_comparison
sbatch compare_trials.run
```

One job scores every checkpoint of every trial on the selection set, then
re-scores each trial's best (and the baselines) on the comparison set — holding
only one dataloader in memory at a time. A trial that fails (missing checkpoint,
OOM, ...) is logged and marked failed but does not abort the rest.

## Outputs

Written under `output_dir`:

- `comparison.csv` — ranked table, one row per trial **and per baseline** (rank,
  `invariance_ratio` on the comparison set, `selection_invariance_ratio` on the
  selection set, within-/between-frame similarity, best checkpoint stage, number
  of checkpoints evaluated, plus each trial's training metadata).
- `comparison_summary.txt` — human-readable ranking and the eval spec (both
  videos, baselines).
- `comparison_invariance_ratio.png` — bar chart, best (lowest) on top.
- `<trial_name>/` — the best checkpoint's within-/between-frame heatmaps + text
  summary (scored on the comparison set), and `checkpoint_scores.csv` (the
  selection-set sweep, selected checkpoint flagged).
- `baseline_<random|imagenet>/` — same heatmaps + summary for each baseline.

A big gap between `selection_invariance_ratio` and `invariance_ratio` for a
trial means its checkpoint choice overfit the selection video — treat its rank
with caution.

All similarity metrics are computed on **mean-centered** pooled features:
pooled CNN features are anisotropic (every cosine similarity is crushed near 1),
so the common mean direction is subtracted first to expose real structure. The
within- vs between-frame heatmaps are the collapse check — if they look the
same, the encoder isn't separating frames; if within is clearly higher, that
gap is the invariance signal (and the invariance ratio is low).

## Direct invocation

```bash
python src/poseforge/pose/contrast/scripts/compute_variant_similarity.py compare \
    --trial-dirs \
        /scratch/.../contrastive_pretraining/trial_20260622_cropped256 \
        /scratch/.../simsiam_pretraining/trial_20260622_simsiam256 \
    --data-dirs /work/.../atomic_batches/BO_Gal4_fly1_trial001 \
    --selection-data-dirs /work/.../atomic_batches/BO_Gal4_fly1_trial002 \
    --baselines random imagenet \
    --atomic-batch-n-samples 32 \
    --atomic-batch-n-variants 2 \
    --batch-size 8 \
    --image-size 912 912 \
    --output-dir /scratch/.../variant_similarity_comparison/comparison_20260625
```

(Full frames, no `--crop-size`. `--batch-size 8` keeps the full-resolution
forward pass within GPU memory; raise it if you have room. Drop
`--selection-data-dirs` to select and rank on the same set, and drop
`--baselines` to skip the untrained references.)

To inspect a single checkpoint instead of comparing trials, use the `single`
subcommand (see the docstring at the top of `compute_variant_similarity.py`).
