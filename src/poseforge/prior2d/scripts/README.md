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
  Always saves its own before/after summary figure. Run separately from
  `filter_student_predictions.sh`, on its output.
- `make_videos.py`: renders annotated videos overlaying raw predictions
  (blue) and, with `--with-ik`, the IK/FK result (green on the same panel,
  plus a second panel with a synthetic 3D view of the IK reconstruction).
  Run once, after `solve_ik.py`, not part of `filter_student_predictions.sh`.

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
