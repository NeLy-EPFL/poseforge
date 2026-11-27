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

Three SLURM scripts: **two extractions** (run once to build the libraries) and
**one generation** (modifies the tiled-inference videos). Edit the paths block at
the top of each, then `sbatch` it.

## 1. Extract the dust library  — `extract_dust.run`

Samples strided aligned frames from every trial into a size-balanced dust library.

```bash
sbatch scripts_on_cluster/add_dust/extract_dust.run
```

Writes `dust_library.npz` + `dust_statistics.json` (+ preview) to `OUTPUT_DIR`.

## 2. Extract the wall library  — `extract_walls.run`

Takes the arena edge from frames whose stage-derived fly-to-wall distance is in
the band (default 1.9–2.5 mm). Needs the arena folder.

```bash
sbatch scripts_on_cluster/add_dust/extract_walls.run
```

Writes `wall_library.npz` + `wall_statistics.json` (+ preview) to `OUTPUT_DIR`.
Tune the band inside the script with
`--distance-config.min-distance-mm` / `--distance-config.max-distance-mm`.

Each trial needs `processed/aligned_behavior_video.mkv`,
`processed/behavior_frames_metadata.csv` and
`processed/behavior_alignment_transforms.h5`. Inspect the previews before
generating.

## 3. Generate the augmented videos  — `generate.run`

Adds dust + walls to every translated video, mirroring the input tree. Runs
`add_dust_all_simulations.py`.

```bash
sbatch scripts_on_cluster/add_dust/generate.run
```

Dust and walls are applied independently at their own measured presence
probabilities. Manual overrides (add to the `python` call in `generate.run`):
`--config.dust-probability <p>` and `--wall-config.wall-probability <p>` for the
rates; `--config.use-measured-distribution False` with
`--config.min-layers`/`--config.max-layers` for a fixed dust level. Omit the wall
library/statistics to add dust only. `--seed` makes the result reproducible;
artifacts are independent per video and per frame.

Compositing is CPU-only. Adjust `--partition` / `--qos` to your cluster if the
defaults here are not valid.
