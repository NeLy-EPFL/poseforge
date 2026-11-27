#!/bin/bash
# Generates one re-postprocess .run per affected trial.
OUTDIR=~/poseforge/scripts_on_cluster/nmf_simulation/batch_scripts_repostprocess
LOGDIR=~/poseforge/scripts_on_cluster/nmf_simulation/logs_repostprocess
mkdir -p "$OUTDIR" "$LOGDIR"

# "trial seg [seg ...]" — your 8 quarantined orphans, grouped by trial
JOBS=(
  "BO_Gal4_fly1_trial002 1"
  "BO_Gal4_fly1_trial003 6"
  "BO_Gal4_fly1_trial004 21"
  "BO_Gal4_fly2_trial001 20"
  "BO_Gal4_fly2_trial003 8"
  "BO_Gal4_fly2_trial004 3"
  "BO_Gal4_fly3_trial001 15 23"
)

for entry in "${JOBS[@]}"; do
  set -- $entry; trial=$1; shift; segs="$*"
  cat > "$OUTDIR/repostproc_${trial}.run" <<EOF
#!/bin/bash -l
#SBATCH --job-name repostproc_${trial}
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 100GB
#SBATCH --time 4:00:00
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --output ${LOGDIR}/repostproc_${trial}_%j.log

. ~/spack/share/spack/setup-env.sh
spack load ffmpeg
conda activate poseforge-flygymv2
cd \$HOME/poseforge

export TRIAL="${trial}"
export SEGS="${segs}"
python - <<'PY'
import os, shutil
from pathlib import Path
from poseforge.neuromechfly.postprocessing import postprocess_segment
NMF = Path("/work/upramdya/stimpfli/poseforge/data/nmf_rendering")
trial = os.environ["TRIAL"]
segs = [int(x) for x in os.environ["SEGS"].split()]
RENDER = ["nmf_sim_render_base.mp4","nmf_sim_render_grayscale.mp4",
          "nmf_sim_render_per_link_color.mp4","nmf_sim_depth.mp4","nmf_sim_segment_id.mp4"]
for sid in segs:
    rec = NMF / trial / f"segment_{sid:03d}"
    if not (rec / "simulation_data.h5").is_file():
        print("SKIP (no raw):", rec); continue
    missing = [f for f in RENDER if not (rec / f).is_file()]
    if missing:
        print("SKIP (missing videos):", rec, missing); continue
    for d in sorted(rec.glob("subsegment_*")):
        if d.is_dir(): shutil.rmtree(d)
    print("re-postprocessing", rec, flush=True)
    postprocess_segment(rec, RENDER, visualize=False,
                        min_subsegment_duration_sec=0.1, use_flybody=False)
print("DONE", trial)
PY
EOF
  echo "wrote repostproc_${trial}.run  (segments: $segs)"
done
