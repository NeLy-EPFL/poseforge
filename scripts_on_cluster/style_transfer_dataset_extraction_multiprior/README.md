# style_transfer_dataset_extraction_multiprior

Builds a CUT training dataset where **domain A carries all geometric priors**,
using `src/poseforge/style_transfer/scripts/extract_dataset_multiprior.py`.

Output layout under `OUTPUT_DIR` (every `trainA*` folder shares file stems, so the
i-th entry of each corresponds to the same frame):

```
trainA/               primary image (grayscale render),      .jpg
trainA_shading/       uniform-albedo shape-from-shading,     .png
trainA_depth/         colorized metric depth,                .png
trainA_normals/       back-projected surface normals,        .png
trainA_segmentation/  body-part segmentation colors,         .png
trainA_mask/          binary silhouette (from the h5),       .png
trainB/               real Spotlight images (no priors),     .jpg
testA/ testA_*/ testB/   validation split, same structure
```

Domain B (real Spotlight) has no synthetic geometry, so only `trainB` / `testB`
images are written.

## Prerequisite — re-rendered simulation data

Each NMF subsegment must contain **all five** buffer videos plus the h5 mask:

- `processed_nmf_sim_render_grayscale.mp4`
- `processed_nmf_sim_render_shading.mp4`
- `processed_nmf_sim_depth.mp4`
- `processed_nmf_sim_normals.mp4`
- `processed_nmf_sim_segment_id.mp4`
- `processed_simulation_data.h5` (`postprocessed/segmentation_labels`)

These only exist for renders produced **after** the flygym-v2 port (commit
`b2ea289`), where `render_depth` and the shading visual are on by default.
Extraction is all-or-nothing per frame: any subsegment missing a buffer is
skipped, so an old render set yields an empty pool. Re-run `nmf_simulation` first
if needed.

## Run

Edit the paths at the top of `job.run` (`NMF_RENDERING_DIR`,
`SPOTLIGHT_RECORDINGS_DIR`, `OUTPUT_DIR`), then:

```bash
sbatch job.run
```

## Pipeline order
1. `nmf_simulation` → renders with geometry buffers
2. `spotlight_unpack_and_categorize` → `spotlight_aligned_and_cropped` (domain B)
3. **this job** → multiprior CUT dataset
4. CUT training (consumes `trainA` + `trainA_<prior>` + `trainB`)
