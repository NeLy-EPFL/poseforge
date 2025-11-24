## v1
- `v1/production_checkpoint_v1.pth` <- `bulk_data/pose_estimation/bodyseg/trial_20251012b/checkpoints/epoch13_step18335.model.pth`
- `v1/production_configs` <- `bulk_data/pose_estimation/bodyseg/trial_20251012b/configs/`
- This version is later replaced due to upstream bug in contrastive pretraining

## v2
- `v2/production_checkpoint.model.pth` <- `bulk_data/pose_estimation/bodyseg/trial_20251118a/checkpoints/epoch14_step12000.model.pth`
- `v2/configs/` <- `bulk_data/pose_estimation/bodyseg/trial_20251118a/configs/`
- This is the retrained checkpoint after the [upstream bug in contrastive pretraining](https://github.com/NeLy-EPFL/poseforge/pull/41) was fixed
- Seems like the validation loss is still going down - might want to train this a bit more in the future
