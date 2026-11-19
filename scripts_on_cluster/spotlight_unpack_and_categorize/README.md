# spotlight_unpack_and_categorize

Unpacks aligned Spotlight recordings into frames and categorizes each frame
(edge / flipped / untracked / rearing), writing the
`spotlight_aligned_and_cropped/<trial>/` layout that the CUT dataset extractor
consumes. Only frames that are **not edge, not flipped, and not rearing** (untracked
kept) are labelled `not flipped` in `predicted_flip_labels.csv`, so the extractor
takes only those for CUT.

## Inputs (on the cluster)
- `$WORK/$USER/poseforge/data/spotlight_new_vid/<trial>/{metadata.zip, processed/}`
- `$WORK/$USER/poseforge/data/spotlight_new_vid/arena72x48/`

## Run

One job per genotype (fans that genotype's trials across the allocated CPUs):

```bash
sbatch job.run G1xCI63
sbatch job.run G213xCI80
sbatch job.run G251xCI80
```

or all at once:

```bash
bash submit_all.sh
```

Output: `$WORK/$USER/poseforge/data/spotlight_aligned_and_cropped/<trial>/` with
`all/` (frame jpg + per-frame `metadata.json`), `predicted_flip_labels.csv`,
`model_prediction/{not_flipped,flipped}/`, `manual_label/`, `categorization_summary.json`.

## Pipeline order
1. **this job** → `spotlight_aligned_and_cropped`
2. `scripts_on_cluster/style_transfer_dataset_extraction/job.run` → CUT dataset
3. CUT training

## Models
- Flip: `production_models/flip_detection/v1` (config `flip_detection`).
- Rearing: `production_models/rearing_detection/v1` (config `rearing_detection`).
- Edge: keypoint-independent corner test (`poseforge.spotlight.valid_frame_filtering`),
  clearance 0.4 mm; flip temporal filter: min run 3, pad 0.3 s.
