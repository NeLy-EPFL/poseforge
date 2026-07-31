# `prior2d` scripts

Pipeline, in order:

- `run_lm_model_on_full_behavior_videos.sh <trial_path>`: runs the landmark
  SLEAP model on one trial's full behavior video, producing a per-trial
  `.slp`/`.h5` of raw predictions.
- `port_slp_labels.py`: combines every trial's raw predictions into one
  aligned-domain `.slp` (no filtering).
- `filter_slp_labels.py`: promotes high-confidence predictions to user
  labels, based on score, leg-segment-length, missing-keypoint, and
  temporal-dedup criteria.
- `convert_slp.py`: bidirectional `.slp` <-> `.h5` conversion, optionally
  rendering an annotated overview video. Bulk arrays (`poses`,
  `instance_scores`, `keypoint_scores`, `accepted`) are gzip-compressed
  datasets; small per-video metadata (`node_names`, `genotypes`,
  `fly_trials`, `n_frames_per_video`) are root `.attrs`.
- (train a "student" SLEAP model on the promoted labels; not scripted here)
- `run_trained_model_on_aligned_videos.sh <trial_path>`: runs the retrained
  **single-instance** student model directly on one trial's aligned video (no
  centroid stage needed). Meant to run locally against a workstation GPU, not
  on the cluster like `run_lm_model_on_full_behavior_videos.sh` above.
- `filter_student_predictions.sh`: runs the student model's predictions
  through the full labeled-data pipeline in one go:
  1. `filter_slp_labels.py` (promote high-confidence predictions, using
     `--min-keypoint-score 0.2` by default)
  2. `convert_slp.py` (`.slp` -> `.h5`, with acceptance flags)
  3. `extract_continuous_periods_from_h5.py` (extract contiguous
     accepted-frame periods into a periods `.h5`, with each period's
     keypoints in both pixel (`pred_2d_px`) and physical mm (`pred_2d_mm`)
     coordinates; always also saves a summary figure of periods per trial
     and length distribution alongside the `.h5`)
- `solve_ik.py`: fits the NeuroMechFly body plan to each period's
  `pred_2d_mm` via QuickIK (trials solved in parallel via joblib), saving a
  re-segmented copy of the periods `.h5` (from step 3 above) with
  `ik_dofangles_rad`, `fk_3d_mm`, and `fk_2d_px`. The root ("thorax") joint
  is always left unobserved (SLEAP's "Th" landmark is a different physical
  point than the body plan's thorax origin, so it's not a valid target for
  it); frames whose worst leg keypoint mismatch exceeds `--max-mismatch`
  (mm) are dropped, splitting or shortening periods as needed, and the three
  new datasets are median-filtered over time (`--filtering-mask-frames`).
  `--neutral-weight` (QuickIK's prior pulling toward the body plan's neutral
  pose) defaults to 0.5; see "Round 4" below for how that was chosen. Always
  saves its own before/after summary figure. Run separately from
  `filter_student_predictions.sh`, on its output.
- `make_videos.py`: renders annotated videos overlaying raw predictions
  (blue) and, with `--with-ik`, the IK/FK result (green on the same panel,
  plus a second panel with a synthetic 3D view of the IK reconstruction).
  Run once, after `solve_ik.py`, not part of `filter_student_predictions.sh`.
- `replay_in_flygym.py`: feeds each period's `ik_dofangles_rad` to a
  leg-actuated `NeuroMechFly` model in FlyGym (CPU physics, not warp) as
  position-actuator targets, one recorded frame per physics step at the
  trial's own recording rate, and renders the resulting simulation from an
  oblique tracking camera and a bottom-up (segmentation-only) camera
  matching the real Spotlight rig. The fly's overall position/orientation is
  not prescribed: it emerges from the physics as the leg actuators push
  against the ground. Also saves an interoperable simulation-data `.h5` and,
  by default, a 4-panel video combining its own renders with
  `make_videos.py`'s 2D pose and synthetic 3D IK/FK panels for the same
  period. See "Round 3" below for the parameters used and a caveat about
  simulated heading drift. `--trial-name <genotype>/<fly_trial>` restricts
  the replay to one trial, and `--n-replay-workers` parallelizes periods via
  joblib (each worker builds its own model); see "Round 5" for the cluster
  array-job setup these two options are for.

Shared code lives in `src/poseforge/prior2d/` (`skeleton_viz.py`,
`geometry.py`, `calibration.py`, `periods.py`), not under `scripts/`, so it
can be imported by more than one script without cross-importing between
scripts.

## Pipeline history

Everything below happened under `bulk_data/prior-2dinvkin/sleap/`, across two
rounds of SLEAP models. The shared upstream files (raw and ported LM
predictions, the retrained model, its raw predictions) are not specific to
any confidence threshold and live under `lm_ported_score_0.5/` only because
that directory is a straight rename of the original (pre-threshold-sweep)
`lm_ported/`; `filter_student_predictions.sh` reads its input `.slp` from
there and writes filtered/periods outputs to the threshold-specific
`lm_ported_score*/` directory instead.

### Round 1: initial "LM" model -> promoted labels -> retrained model

1. `run_lm_model_on_full_behavior_videos.sh <trial_path>`, once per trial,
   using the existing two-stage LM model
   (`250327_002024.centroid` + `260701_180342.centered_instance.n=215`).
   Writes each trial's own `sleap/prediction_lm_full_behavior_video.slp`.
2. `port_slp_labels.py` combined all 63 trials' raw LM predictions into one
   aligned-domain file, `lm_ported_v000.slp` (1,247,966 labeled frames, 37
   nodes, unfiltered).
3. `filter_slp_labels.py` promoted the confident predictions in
   `lm_ported_v000.slp` to user labels (score/leg-length/missing-keypoint/
   dedup criteria), producing the training data for a new model.
4. A new **single-instance** SLEAP model (a different architecture from the
   original two-stage centroid+centered-instance LM model) was trained on
   those promoted labels: `models/lmport_v000_run000.single_instance.n=467875/`
   (`best.ckpt`, saved 2026-07-29 19:35; `n=467875` labeled points across
   train+val).
5. The new model's predictions were intended to land in
   `lm_ported_v000_trained000.slp` (created 2026-07-30 11:51).

   **Caveat, found while answering a verification question this session**:
   `lm_ported_v000_trained000.slp` is value-identical to `lm_ported_v000.slp`
   -- checked exhaustively (every frame, not a sample) for two full videos
   (20,425 and 20,790 frames), zero differing keypoints. So despite the name,
   this file does not currently reflect the retrained model's predictions;
   it appears to just carry the original LM model's predictions forward. The
   only file that looks like genuine output from the new model is
   `predictions/aligned_behavior_video.mkv_lm_ported_v000.slp.260730_050224.predictions.slp`,
   but it holds only a single labeled frame (frame 12013 of
   `G-213xCI55_260720/fly000_trial000`) -- consistent with a quick load/sanity
   check of the new model rather than a real inference run. Whether the
   intended full-scale re-inference with the new model was ever completed (and
   if so, where its output went) is unresolved; every step from
   `filter_student_predictions.sh` onward below has been run against
   `lm_ported_v000_trained000.slp` as found, i.e. effectively against the
   original LM model's predictions, not the retrained model's.

   These files all live under `lm_ported_score_0.5/` (see note above).

### Round 2: filtering, periods, IK (this session, 2026-07-30)

Run once with `--min-keypoint-score 0.5` (outputs under
`lm_ported_score_0.5/`), then again with the new default `0.2` (outputs
under `lm_ported_score/`), which keeps ~30% more frames after IK filtering
with no change in fit quality.

```bash
source_dir="bulk_data/prior-2dinvkin/sleap/lm_ported_score_0.5"
output_dir="bulk_data/prior-2dinvkin/sleap/lm_ported_score"
scripts_dir="src/poseforge/prior2d/scripts"

# filter_slp_labels.py -> convert_slp.py -> extract_continuous_periods_from_h5.py
# (promotes confident frames in lm_ported_v000_trained000.slp to labels, converts
# to .h5 with acceptance flags, extracts contiguous accepted periods, and always
# saves its own summary figure)
bash "$scripts_dir/filter_student_predictions.sh"

# Fits the NeuroMechFly body plan via QuickIK to each period's pred_2d_mm.
# Iterated on solver parameters this session (final values shown):
# - neutral_weight raised 10x over the QuickIK tutorial's own 10x-of-default
#   (i.e. 0.1, 100x SolverConfig's bare default).
# - ThC (leg-base) observation weight halved (0.5x), since it's barely
#   observed independently of the root pose it's rigidly close to.
# - Root ("thorax") joint always left unobserved: SLEAP's "Th" landmark and
#   the body plan's thorax origin are different physical points.
# - Frames whose worst leg keypoint (excluding "Th") mismatch exceeds 0.3mm
#   are dropped (--max-mismatch 0.3), splitting/shortening periods as needed;
#   filtering (median, 5-frame window) happens before this check so the
#   threshold bounds what's actually stored.
python "$scripts_dir/solve_ik.py" \
    --periods-path "$output_dir/periods.h5" \
    --output-path "$output_dir/periods_ikfk.h5" \
    --n-jobs -1

# Renders annotated videos: kchain_plotting_colors per leg, thin white raw
# predictions on the left (2D) panel, plus a right panel with a synthetic 3D
# view of the IK reconstruction (yaw-tracking camera, 1mm ground-plane grid).
# Encodes on GPU (NVENC) via pvio, 3 videos in parallel via joblib.
python "$scripts_dir/make_videos.py" \
    "$output_dir/periods_ikfk.h5" \
    "$output_dir/period_videos" \
    --with-ik \
    --periods-per-trial 1
```

### Round 3: FlyGym replay (this session, 2026-07-31)

```bash
output_dir="bulk_data/prior-2dinvkin/sleap/lm_ported_score"
scripts_dir="src/poseforge/prior2d/scripts"

# Replays each period's ik_dofangles_rad in FlyGym (CPU) and renders the
# result; also builds a 4-panel video combining its own tracking-cam/segid
# renders with make_videos.py's 2D pose and synthetic 3D IK/FK panels for
# the same period (--combine-panels, on by default). Position-actuator
# gain (kp), leg-adhesion gain, and the segid crop/zoom fraction are
# hardcoded constants in the script (ACTUATOR_GAIN=50, ADHESION_GAIN=0.3,
# ZOOM_CROP_FRACTION=0.55), tuned this session against visible jerkiness on
# noisy IK frames rather than exposed as CLI flags.
#
# Caveat found this session: since the IK never solves for the root's own
# orientation (only relative leg joint angles), the simulated fly's heading
# is entirely emergent from leg-actuator reaction forces, and can drift
# noticeably from the real recorded fly's heading over a period (one period
# checked: ~25 deg simulated drift vs. ~2 deg in the real recording) even
# though every leg joint faithfully tracks its recorded target angle. Only
# the combined video's bottom-left segid panel is rotated frame-by-frame
# (using the sim's own recorded heading) to counteract this for viewing;
# the standalone `_segid.mp4` stays a raw, unrotated render, and the
# interoperable h5 data is left as simulated, unrotated, in all cases.
python "$scripts_dir/replay_in_flygym.py" \
    "$output_dir/periods_ikfk.h5" \
    "$output_dir/flygym_replays" \
    --periods-per-trial 1
```

### Round 4: genuine retrained-model predictions, neutral_weight=0.5 (this session, 2026-07-31)

Round 1 found that `lm_ported_v000_trained000.slp` was actually the original LM
model's predictions carried forward, not the retrained single-instance
model's. This round ran genuine inference with that model and repeated
Rounds 2-3 on the result, in a new `lm_ported_v001/` (no threshold suffix:
only one `--min-keypoint-score` was tried this time).

Separately, `solve_ik.py`'s `--neutral-weight` (QuickIK's prior pulling
toward the body plan's neutral pose) was swept over 0.1 (the prior default)
/0.5/1.0 by comparing rendered FlyGym replays; 0.5 was chosen (stronger than
0.1, but 1.0 visibly over-smoothed genuine leg motion) and is now
`solve_ik.py`'s own default, used below.

```bash
data_root="/mnt/upramdya_data/VAS/poseforge_paper_data"
new_root="bulk_data/prior-2dinvkin/sleap/lm_ported_v001"
scripts_dir="src/poseforge/prior2d/scripts"

# Runs the retrained single-instance model directly on each trial's aligned
# video (no centroid stage), one trial at a time, looping locally over every
# trial in lm_ported_v000.slp (~2 min/trial on a local GPU, ~2h11m total for
# 63 trials -- no cluster/manifest needed for this step).
for trial_dir in $(python -c "
import sleap_io as sio
from pathlib import Path
labels = sio.load_slp('bulk_data/prior-2dinvkin/sleap/lm_ported_score_0.5/lm_ported_v000.slp')
for v in labels.videos:
    print(Path(v.filename).parent.parent)
"); do
    bash "$scripts_dir/run_trained_model_on_aligned_videos.sh" "$trial_dir"
done

# Combines every trial's sleap/prediction_trained000_aligned.{slp,h5} into
# one aligned-domain .slp (--aligned-input: no per-frame transform needed).
python "$scripts_dir/port_slp_labels.py" \
    --output-path "$new_root/lm_ported_v001.slp" \
    --aligned-input \
    --h5-relpath sleap/prediction_trained000_aligned.h5 \
    --reference-slp-relpath sleap/prediction_trained000_aligned.slp

# filter_slp_labels.py -> convert_slp.py -> extract_continuous_periods_from_h5.py,
# by hand (not filter_student_predictions.sh, whose paths are hardcoded to
# lm_ported_score_0.5/lm_ported_score), same criteria as Round 2.
python "$scripts_dir/filter_slp_labels.py" \
    --input-path "$new_root/lm_ported_v001.slp" \
    --output-path "$new_root/filtered.slp" \
    --aligned-input \
    --min-keypoint-score 0.2 \
    --max-leg-segment-length 200.0 \
    --max-missing-keypoints 0 \
    --min-pose-change 2.0
python "$scripts_dir/convert_slp.py" \
    --slp2h5 \
    --input-path "$new_root/filtered.slp" \
    --output-path "$new_root/filtered.h5" \
    --include-acceptance
python "$scripts_dir/extract_continuous_periods_from_h5.py" \
    --input-path "$new_root/filtered.h5" \
    --output-path "$new_root/periods.h5"

# solve_ik.py (now defaulting to --neutral-weight 0.5) -> make_videos.py ->
# replay_in_flygym.py, same as Rounds 2-3.
python "$scripts_dir/solve_ik.py" \
    --periods-path "$new_root/periods.h5" \
    --output-path "$new_root/periods_ikfk.h5" \
    --n-jobs -1
python "$scripts_dir/make_videos.py" \
    "$new_root/periods_ikfk.h5" \
    "$new_root/period_videos" \
    --with-ik \
    --periods-per-trial 1
python "$scripts_dir/replay_in_flygym.py" \
    "$new_root/periods_ikfk.h5" \
    "$new_root/flygym_replays" \
    --periods-per-trial 1
```

### Round 5: cluster-parallel FlyGym replay, all periods (this session, 2026-07-31)

`replay_in_flygym.py` gained two options for running the full (not just
longest-per-trial) replay on a cluster, one SLURM array task per trial:

- `--trial-name <genotype>/<fly_trial>` restricts the run to one trial. With
  it, `output_dir` is no longer cleared at startup (only a full,
  every-trial run clears it), since it's expected to be one of many
  concurrent per-trial array tasks sharing one `output_dir`.
- `--n-replay-workers` parallelizes periods within a trial via joblib (the
  only parallelism in the script; each worker's own video/H5 writing runs
  synchronously as part of its period, not further parallelized). MuJoCo
  objects aren't picklable across processes, so each worker builds its own
  `NeuroMechFly` model once (cached for every period joblib routes to that
  worker), rather than sharing the main process's.
- `--periods-per-trial` now defaults to `-1`, replaying every period in the
  trial (previously the default was `1`, the longest period only; pass `1`
  explicitly for that).

Progress logging for long (all-periods, many-trial) runs: a `"queued N
period replays"` line per trial, a `"Progress: i/N periods done"` line per
period with `--n-replay-workers 1`, and joblib's own `"Done i out of N |
elapsed / remaining"` lines otherwise.

The manifest (one `<genotype>/<fly_trial>` per line, matching `--trial-name`)
was generated from `periods.h5` (stable across `solve_ik.py` reruns, unlike
`periods_ikfk.h5`):

```bash
python -c "
import h5py
with h5py.File('bulk_data/prior-2dinvkin/sleap/lm_ported_v001/periods.h5', 'r') as f:
    for genotype in f:
        for fly_trial in f[genotype]:
            print(f'{genotype}/{fly_trial}')
" > scripts_on_cluster/prior2d/flygym_replay/manifest.txt
```

`scripts_on_cluster/prior2d/flygym_replay/run_flygym_replay.run` (SLURM
array job, 36 cpus/128GB/4h per task, 18 replay workers) and
`submit_flygym_replay.sh` (`sbatch --array=1-N`) submit one task per
manifest line. The cluster is CPU-only, so the job sets
`MUJOCO_GL=osmesa` (MuJoCo's default EGL renderer needs a GPU; OSMesa is
the software-rendering fallback) -- `replay_in_flygym.py` itself has no
GPU dependency otherwise (`write_video` and FlyGym's own `Renderer.save_video`
already use CPU libx264; `pvio.read_frames_from_video`, used by
`--combine-panels` to read the source aligned video, is decode-only). Only
tested locally (this workstation has a GPU, so `MUJOCO_GL=osmesa` itself
could not be exercised here); confirm OSMesa is installed on the cluster's
compute nodes before submitting.
