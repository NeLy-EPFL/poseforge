# Rearing detection model

Classifies an aligned Spotlight head crop as **rearing** (head lifted, antennae
hidden) vs **not_rearing** (antennae clearly visible). Used by
`poseforge.spotlight.scripts.unpack_and_categorize_spotlight` to drop rearing frames
from the CUT training set.

## Architecture
- MobileNetV3-small (ImageNet-pretrained), grayscale replicated to 3 channels and
  ImageNet-normalized, 2-class head. See `poseforge.spotlight.rearing_detection`.
- Input: a fixed head crop of the aligned frame — `half=130` (260 px) around the
  median neck, shifted up by `y_offset=-60`, resized to `input_size=160`.
- The crop window is **fixed per trial** (median neck in the aligned frame), so
  frames with missing pose keypoints are still scored (tracking often fails *because*
  the fly is rearing).

## Checkpoint format (`v1/production_checkpoint_v1.pth`)
A dict with `snapshots` (list of 6 epoch `state_dict`s — predictions are averaged
over them as a snapshot ensemble, which calibrates the small-data model), `crop`
(`half`, `y_offset`, `input_size`), `class_order` (`["not_rearing", "rearing"]`,
index 1 = rearing), `backbone`, and `training` metadata.

## Training
- 1,194 hand labels across 8 fly-trials / 3 genotypes (G1xCI63, G213xCI80, G251xCI80),
  old and young.
- 5-fold out-of-fold: **AUC 0.936, accuracy 0.874** at threshold 0.5.
- Augmentation: anisotropic one-axis scale, rotation ≤10°, horizontal mirror.
