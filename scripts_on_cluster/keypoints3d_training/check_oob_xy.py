import sys, glob, h5py, numpy as np
from collections import Counter

# Keypoint order matches poseforge.neuromechfly.constants.keypoint_segments_nmf
JOINTS = ["coxa", "trochanterfemur", "tibia", "tarsus1", "tarsus5"]
LEGS = [s + p for s in "lr" for p in "fmh"]  # lf, lm, lh, rf, rm, rh
KP = [f"{leg}_{j}" for leg in LEGS for j in JOINTS] + ["l_pedicel", "r_pedicel"]

for p in sys.argv[1:]:
    for f in sorted(glob.glob(f"{p}/**/atomicbatch*_labels.h5", recursive=True)):
        xy = h5py.File(f)["keypoint_pos"][:, :, :2]  # (n_frames, n_kp, 2)
        oob = (xy > 912) | (xy < 0)
        if not oob.any():
            continue
        fr, kp, ax = np.where(oob)
        # Aggregate by segment (leg) / subsegment (joint)
        seg_sub = [
            (KP[k].split("_", 1) if k < len(KP) else (f"kp{k}", "?"))
            for k in kp
        ]
        by_kp = Counter(f"{s}_{sub}" for s, sub in seg_sub)
        by_seg = Counter(s for s, _ in seg_sub)
        by_sub = Counter(sub for _, sub in seg_sub)
        print(
            f"{f}\n"
            f"  oob_count={int(oob.sum())}  max={xy.max():.1f}  min={xy.min():.1f}\n"
            f"  by segment   : {dict(by_seg.most_common())}\n"
            f"  by subsegment: {dict(by_sub.most_common())}\n"
            f"  top keypoints: {dict(by_kp.most_common(8))}"
        )
