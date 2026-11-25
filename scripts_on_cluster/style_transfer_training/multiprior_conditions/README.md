# multiprior_conditions

Four CUT training conditions on the multiprior dataset — 2 cropping regimes x 2
buffer sets. Everything else is identical, so only the regime and buffers vary.

| regime | buffers | `--preprocess` | `--domainA_buffers` |
|--------|---------|----------------|---------------------|
| randomcrop | imageonly | `random_center_patch` | `image` |
| randomcrop | allpriors | `random_center_patch` | `image,shading,depth,normals,segmentation` |
| curriculum | imageonly | `resize_and_crop` | `image` |
| curriculum | allpriors | `resize_and_crop` | `image,shading,depth,normals,segmentation` |

- **randomcrop**: 256 patches taken straight from the native frame throughout.
- **curriculum**: full frame for the first `N_EPOCHS` (100), then zoomed-in crops
  during the decay tail (`finetune_load_size` > `crop_size`).

## Run
```bash
bash submit_all.sh
```
or one condition:
```bash
sbatch --job-name mp_curriculum_allpriors job.run curriculum allpriors
```

## Notes
- Set `CAMPAIGN_TAG` and `DATASET_PATH` at the top of `job.run`.
  Checkpoints / logs / wandb go under `$WORK/$USER/poseforge/style_transfer/<CAMPAIGN_TAG>/`.
- The SLURM job log is written into **this folder** as
  `<job name>_<YYYYmmdd_HHMMSS>.out` (via an `exec` redirect, since `#SBATCH`
  can't build a name with the current time).
- Requires the updated CUT + poseforge code pulled on the cluster and the
  multiprior dataset built at `DATASET_PATH`.
