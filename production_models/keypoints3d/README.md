## v1
- `v1/production_checkpoint_v1.pth` <- `bulk_data/pose_estimation/keypoints3d/trial_20251102a/checkpoints/epoch28_step9167.model.pth`
- `v1/configs` <- `bulk_data/pose_estimation/keypoints3d/trial_20251102a/configs/`
- This version is yanked because the results were not so good. In this version, the 2D pose heatmap is predicted at a resolution of 64x64 (stride of 4 compared to the input size of 256x256). Increasing the prediction resolution to 128x128 (stride=2) led to an improvement.
- This version is later double-yanked due to the discovery of an [upstream bug in contrastive pretraining](https://github.com/NeLy-EPFL/poseforge/pull/41)

## v2
- `v2/production_checkpoint.model.pth` <- `bulk_data/pose_estimation/keypoints3d/trial_20251118a/checkpoints/epoch19_step9167.model.pth`
- `v2/configs/` <- `bulk_data/pose_estimation/keypoints3d/trial_20251118a/configs/`
- Compared to v1, this version:
  - predicts 2D pose heatmap at stride=2
  - starts from a feature extractor checkpoint with the upstream contrastive pretraining bug fixed.
