# Add dust and walls to style-transferred videos

Post-processing step that runs **after** the tiled style-transfer inference. CUT
produces clean translated videos; this step writes a parallel directory tree in
which every frame has realistic imaging artifacts added:

- **Dust / fibers / specks** — sampled from a library of **primitives** extracted
  from real recordings, each randomly rotated, scaled, defocused and additively
  blended over the whole frame (so some lands on the fly, as real foreground
  debris does). Edge-touching primitives are dropped at extraction.
- **Arena walls** — the bright chamber edge. A frame is used as a wall source
  only when the fly's **stage-derived distance to the arena wall** is in a chosen
  band (default **1.9–2.5 mm**: close enough that the edge is in view, far enough
  that it does not sit on the fly). The fly is removed and the residual full frame
  is kept; at compositing time the wall is **added additively** — just like dust —
  after mirroring on both axes only (no masking).

Dust and walls are applied **independently**, each at its own **measured presence
probability**: the fraction of real frames with dust (`dust_statistics.json`) and
the fraction whose stage distance puts a wall in view (`wall_statistics.json`).
Dust amount and sizes also follow the measured distribution.

## 1. Build the libraries + statistics (one-time)

Dust and walls are extracted **separately**. Dust is sampled from strided aligned
frames; walls come from the stage-distance band (needs the arena folder):

```bash
python -m poseforge.dust.scripts.extract_dust \
    --dataset-root /path/to/261117_poseforgetraining \
    --output-dir   /work/upramdya/stimpfli/poseforge/dust
```

```bash
python -m poseforge.dust.scripts.extract_walls \
    --dataset-root /path/to/261117_poseforgetraining \
    --arena-dir    /path/to/261117_poseforgetraining/arena72x48 \
    --output-dir   /work/upramdya/stimpfli/poseforge/dust
```

`extract_dust` writes `dust_library.npz` + `dust_statistics.json`; `extract_walls`
writes `wall_library.npz` + `wall_statistics.json` (both with preview sheets).
Each trial needs `processed/aligned_behavior_video.mkv`,
`processed/behavior_frames_metadata.csv` and
`processed/behavior_alignment_transforms.h5`. Tune the wall band with
`--distance-config.min-distance-mm` / `--distance-config.max-distance-mm`.
Inspect the previews before running.

## 2. Add artifacts to all simulations

Single run:

```bash
python scripts_on_cluster/add_dust/add_dust_all_simulations.py \
    --input-basedir  /work/.../tiled_translated_videos \
    --output-basedir /work/.../tiled_translated_videos_dusty \
    --dust-library-path    /work/.../dust/dust_library.npz \
    --dust-statistics-path /work/.../dust/dust_statistics.json \
    --wall-library-path    /work/.../dust/wall_library.npz \
    --wall-statistics-path /work/.../dust/wall_statistics.json \
    --no-override
```

The output tree mirrors the input tree. Amounts follow the measured distributions
by default. Manual overrides: `--config.dust-probability <p>` and
`--wall-config.wall-probability <p>` for the presence rates;
`--config.use-measured-distribution False` with
`--config.min-layers/--config.max-layers` for a fixed dust level. Omit the wall
library/statistics to add dust only. `--seed` makes the result reproducible;
artifacts are independent per video and per frame.

## 3. As a batch job

Edit the paths at the top of `gen_batch_scripts.py`, then:

```bash
cd scripts_on_cluster/add_dust
python gen_batch_scripts.py   # writes batch_scripts/*.run
bash submit_all.sh            # submits them
```

Dust compositing is CPU-only, so `template.run` requests no GPU. Adjust
`--partition` if your cluster's CPU partition is not named `cpu`.
