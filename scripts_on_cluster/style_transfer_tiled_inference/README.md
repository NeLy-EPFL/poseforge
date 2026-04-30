# Style Transfer Tiled Inference

Batch processing scripts for running tiled style transfer inference on all simulations in a dataset.

## Features

- **Auto-discovery**: Automatically finds and processes all simulation directories under a base path.
- **Randomized seams**: By default, uses `--randomize_seams` and `--weight_type cosine` for smooth blending.
- **Configurable data sources**: Easily change the base simulation folder and input/output video filenames.
- **Per-checkpoint jobs**: Generate separate Slurm jobs for different model checkpoints and data sources.

## Usage

### 1. Configure Jobs

Edit `gen_batch_scripts.py` to specify:
- Checkpoint paths
- Simulation base directories (e.g., `bulk_data_flybody/` or `bulk_data/`)
- Output base directories
- Input video filenames (e.g., `processed_nmf_sim_render_flybody_grayscale.mp4`)
- Job names

Example:
```python
jobs = [
    (
        "/path/to/checkpoint/765_net_G.pth",
        "bulk_data_flybody/",
        "bulk_data/style_transfer/production/tiled_translated_videos/flybody/",
        "processed_nmf_sim_render_flybody_grayscale.mp4",
        "flybody_gray_765"
    ),
]
```

### 2. Generate Batch Scripts

```bash
cd ~/poseforge
python scripts_on_cluster/style_transfer_tiled_inference/gen_batch_scripts.py
```

This generates `.run` files in `batch_scripts/`.

### 3. Submit Jobs

```bash
cd ~/poseforge/scripts_on_cluster/style_transfer_tiled_inference/batch_scripts
sbatch *.run
```

## Default Parameters

- `weight_type`: `cosine` (smooth blending at tile boundaries)
- `randomize_seams`: `True` (different seam pattern per frame)
- `debug_mode`: `False` (set to `True` in template to visualize tile grid)
- `device`: `cuda` (GPU inference)

## Customization

Edit `template.run` to change:
- `weight_type` (e.g., `gaussian`, `pyramid`, `uniform`)
- `randomize_seams` (remove the flag to disable)
- `--seed` (add for reproducible randomization)
- Resource allocation (nodes, cpus, memory, time, partition, qos)

## Direct Invocation

You can also run the inference script directly:

```bash
python scripts_on_cluster/style_transfer_tiled_inference/run_tiled_inference_all_simulations.py \
    --checkpoint-path /path/to/765_net_G.pth \
    --simulations-basedir bulk_data_flybody/ \
    --output-basedir bulk_data/style_transfer/production/tiled_translated_videos/flybody/ \
    --input-video-filename processed_nmf_sim_render_flybody_grayscale.mp4 \
    --weight-type cosine \
    --randomize-seams \
    --device cuda \
    --verbose
```
