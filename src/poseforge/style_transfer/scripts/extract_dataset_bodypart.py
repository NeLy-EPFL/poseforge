import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from pvio.io import read_frames_from_video
from tqdm import tqdm

try:
    from poseforge.style_transfer.scripts.extract_dataset import (
        list_nmf_simulations_and_num_frames,
    )
except ModuleNotFoundError:
    # Allows running this file directly as a script from this directory.
    from extract_dataset import list_nmf_simulations_and_num_frames


TARGET_SEGMENT_NAMES = [
    "lf_coxa",
    "lf_trochanterfemur",
    "lf_tibia",
    "lf_tarsus1",
    "lm_coxa",
    "lm_trochanterfemur",
    "lm_tibia",
    "lm_tarsus1",
    "lh_coxa",
    "lh_trochanterfemur",
    "lh_tibia",
    "lh_tarsus1",
    "rf_coxa",
    "rf_trochanterfemur",
    "rf_tibia",
    "rf_tarsus1",
    "rm_coxa",
    "rm_trochanterfemur",
    "rm_tibia",
    "rm_tarsus1",
    "rh_coxa",
    "rh_trochanterfemur",
    "rh_tibia",
    "rh_tarsus1",
]

END_JOINTS = ["rf_tarsus5", "rm_tarsus5", "rh_tarsus5", "lf_tarsus5", "lm_tarsus5", "lh_tarsus5"]

# For each segment, distal endpoint is taken from this child segment's proximal point.
# If absent, distal == proximal.
DISTAL_CHILD_BY_SEGMENT = {}
for leg in ["lf", "lm", "lh", "rf", "rm", "rh"]:
    DISTAL_CHILD_BY_SEGMENT[f"{leg}_coxa"] = f"{leg}_trochanterfemur"
    DISTAL_CHILD_BY_SEGMENT[f"{leg}_trochanterfemur"] = f"{leg}_tibia"
    DISTAL_CHILD_BY_SEGMENT[f"{leg}_tibia"] = f"{leg}_tarsus1"
    DISTAL_CHILD_BY_SEGMENT[f"{leg}_tarsus1"] = f"{leg}_tarsus5"

CANONICAL_TO_NMF_SEGMENT = {
    "ThC": "coxa",
    "CTr": "trochanterfemur",
    "FTi": "tibia",
    "TiTa": "tarsus1",
    "Claw": "tarsus5",
}

EXCLUDED_PRED_NAMES = ["RPedicel", "LPedicel"]


def convert_prediction_name_to_target(pred_name: str) -> str:
    """Map keypoint names from prediction files to TARGET_SEGMENT_NAMES.

    This function is intentionally centralized so custom naming rules can be
    added in one place.
    """
    if isinstance(pred_name, bytes):
        pred_name = pred_name.decode("utf-8")
    name = pred_name.strip().replace("-", "_")

    if name in TARGET_SEGMENT_NAMES:
        return name

    # Canonical leg form from keypoints3d model, e.g. LFCoxa, RMTiTa
    m = re.match(r"^([LR][FMH])(ThC|CTr|FTi|TiTa|Claw)$", name)
    if m:
        leg, canonical = m.groups()
        return f"{leg.lower()}_{CANONICAL_TO_NMF_SEGMENT[canonical]}"
    
    return name.lower()


@dataclass
class SpotlightTrialInfo:
    trial_dir: Path
    aligned_video_path: Path
    prediction_h5_path: Path
    usable_frame_ids: np.array


def list_spotlight_recordings_and_usable_frames(
    spotlight_data_dir: Path,
) -> dict[Path, SpotlightTrialInfo]:
    print("Indexing usable frames from Spotlight recordings:")
    trial_info = {}

    for trial_dir in sorted(spotlight_data_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        
        print(trial_dir)
        aligned_video_path = trial_dir / "processed/aligned_behavior_video.mkv"
        usable_csv = trial_dir / "poseforge_output/usable_frames.csv"
        pred_h5 = trial_dir / "poseforge_output/keypoints3d_prediction.h5"

        if not aligned_video_path.is_file() or not usable_csv.is_file() or not pred_h5.is_file():
            logging.warning(
                f"Skipping {trial_dir.name}: expected processed/aligned_behavior_video.mkv and poseforge_output files"
            )
            continue

        df_usable = pd.read_csv(usable_csv)
        if "usable" not in df_usable.columns or "behavior_frameid" not in df_usable.columns:
            raise ValueError(
                f"Expected columns ['usable', 'behavior_frameid'] in {usable_csv}; "
                f"found {df_usable.columns.tolist()}"
            )

        with h5py.File(pred_h5, "r") as f:
            if "conf_xy" not in f:
                raise ValueError(f"Missing 'conf_xy' dataset in {pred_h5}")
            conf_xy = f["conf_xy"][:]  # (n_frames, n_keypoints)
        min_conf_per_frame = conf_xy.min(axis=1)

        frame_ids = df_usable["behavior_frameid"].to_numpy(dtype=int)
        if (frame_ids < 0).any() or (frame_ids >= len(min_conf_per_frame)).any():
            raise IndexError(
                f"behavior_frameid in {usable_csv} contains out-of-range indices for {pred_h5} "
                f"(n_frames={len(min_conf_per_frame)})"
            )

        usable_csv_mask = df_usable["usable"].to_numpy(dtype=bool)
        conf_mask = min_conf_per_frame[frame_ids] >= 0.5
        final_usable_mask = usable_csv_mask & conf_mask

        usable_indices = df_usable.loc[final_usable_mask, "behavior_frameid"].to_numpy(dtype=int)

        trial_info[trial_dir] = SpotlightTrialInfo(
            trial_dir=trial_dir,
            aligned_video_path=aligned_video_path,
            prediction_h5_path=pred_h5,
            usable_frame_ids=usable_indices,
        )
        print(f"  {trial_dir}: {len(usable_indices)} usable frames")

    return trial_info


def build_endpoints(
    target_segment_names: list[str],
    xy_lookup: dict[str, np.ndarray],
    is_spotlight: bool = False,
) -> np.ndarray:
    if is_spotlight:
        SCALING = 900.0/256.0
    else:
        SCALING = 1.0
    k = len(target_segment_names)
    xy = np.full((k, 2, 2), np.nan, dtype=np.float32)

    for seg_idx, seg_name in enumerate(target_segment_names):
        prox_xy = xy_lookup[seg_name]
        distal_seg = DISTAL_CHILD_BY_SEGMENT[seg_name]
        dist_xy = xy_lookup[distal_seg]
        xy[seg_idx, 0, :] = prox_xy * SCALING
        xy[seg_idx, 1, :] = dist_xy * SCALING

    return xy


@dataclass
class AnnotationSample:
    rel_path: str
    xy: np.ndarray
    image_hw: tuple[int, int]
    source_path: str


class AnnotationAccumulator:
    def __init__(self, split_name: str, domain_name: str, segment_names: list[str]):
        self.split_name = split_name
        self.domain_name = domain_name
        self.segment_names = segment_names
        self.samples: list[AnnotationSample] = []

    def add(self, sample: AnnotationSample) -> None:
        self.samples.append(sample)

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self.samples)
        k = len(self.segment_names)

        rel_paths = np.array([s.rel_path for s in self.samples], dtype=np.str_)
        xy = np.full((n, k, 2, 2), np.nan, dtype=np.float32)
        image_hw = np.zeros((n, 2), dtype=np.int32)
        source_paths = np.array([s.source_path for s in self.samples], dtype=np.str_)

        for i, s in enumerate(self.samples):
            xy[i] = s.xy
            image_hw[i] = np.array(s.image_hw, dtype=np.int32)

        split_name = np.array([self.split_name] * n, dtype=np.str_)
        domain_name = np.array([self.domain_name] * n, dtype=np.str_)

        np.savez_compressed(
            output_path,
            segment_names=np.array(self.segment_names, dtype=np.str_),
            rel_paths=rel_paths,
            xy=xy,
            image_hw=image_hw,
            split_name=split_name,
            domain_name=domain_name,
            source_paths=source_paths,
        )
        print(f"Saved {output_path} with {n} items")


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    return image


def frame_xy_vis_from_synthetic(
    camera_coords: np.ndarray,
    segment_names_in_h5: list[str],
) -> dict[str, np.ndarray]:
    xy_lookup = {}
    for idx, seg_name in enumerate(segment_names_in_h5):
        xy = camera_coords[idx, :2].astype(np.float32)
        if not np.isfinite(xy).all():
            raise ValueError(f"Non-finite synthetic keypoint coordinates for '{seg_name}'")
        xy_lookup[seg_name] = xy
    return xy_lookup


def frame_xy_vis_from_prediction(
    pred_xy_frame: np.ndarray,
    pred_names: list[str],
) -> dict[str, np.ndarray]:
    xy_lookup = {}

    for idx, raw_name in enumerate(pred_names):
        if raw_name in EXCLUDED_PRED_NAMES:
            continue
        mapped_name = convert_prediction_name_to_target(raw_name)
        if not (mapped_name in TARGET_SEGMENT_NAMES or mapped_name in END_JOINTS):
            raise ValueError(
                f"Mapped keypoint '{mapped_name}' (from raw '{raw_name}') is not in TARGET_SEGMENT_NAMES"
            )
        xy = pred_xy_frame[idx, :2].astype(np.float32)
        if not np.isfinite(xy).all():
            raise ValueError(
                f"Non-finite spotlight keypoint coordinates for mapped keypoint '{mapped_name}'"
            )
        if mapped_name not in xy_lookup:
            xy_lookup[mapped_name] = xy
        else:
            raise ValueError(f"Duplicate keypoint name after mapping: {mapped_name}")

    return xy_lookup


def extract_synthetic_with_annotations(
    nmf_specs: list[tuple[Path, int, str]],
    output_dir: Path,
    accumulators: dict[str, AnnotationAccumulator],
) -> None:
    print("Extracting synthetic images + annotations...")
    specs_by_video = defaultdict(list)
    for video_path, frame_idx, split_dir_name in nmf_specs:
        specs_by_video[video_path].append((frame_idx, split_dir_name))

    for video_path, specs in tqdm(specs_by_video.items(), disable=None):
        specs = sorted(specs, key=lambda item: item[0])
        frame_indices = [idx for idx, _ in specs]
        frames, _fps = read_frames_from_video(video_path, frame_indices)
        frames_by_idx = {idx: frame for idx, frame in zip(frame_indices, frames)}

        sim_data_path = video_path.parent / "processed_simulation_data.h5"
        with h5py.File(sim_data_path, "r") as f:
            camera_coords_ds = f["postprocessed/keypoint_pos/camera_coords"]
            seg_names = camera_coords_ds.attrs["keys"].tolist()

            for frame_idx, split_dir_name in specs:
                frame = ensure_rgb(frames_by_idx[frame_idx])
                img_h, img_w = frame.shape[:2]

                trial, segment_id, subsegment_id = video_path.parent.parts[-3:]
                image_filename = f"{trial}_{segment_id}_{subsegment_id}_frame_{frame_idx:06d}.jpg"

                split_dir = output_dir / split_dir_name
                split_dir.mkdir(parents=True, exist_ok=True)
                output_image_path = split_dir / image_filename
                imageio.imwrite(output_image_path, frame)

                frame_coords = camera_coords_ds[frame_idx]  # (K, 3)
                xy_lookup = frame_xy_vis_from_synthetic(frame_coords, seg_names)
                xy = build_endpoints(TARGET_SEGMENT_NAMES, xy_lookup, False)

                rel_path = str(output_image_path.relative_to(output_dir))
                sample = AnnotationSample(
                    rel_path=rel_path,
                    xy=xy,
                    image_hw=(img_h, img_w),
                    source_path=str(video_path),
                )
                accumulators[split_dir_name].add(sample)


def extract_spotlight_with_annotations(
    spotlight_specs: list[tuple[Path, int, str]],
    output_dir: Path,
    trial_info_by_dir: dict[Path, SpotlightTrialInfo],
    accumulators: dict[str, AnnotationAccumulator],
) -> None:
    print("Extracting spotlight images + annotations...")

    specs_by_trial = defaultdict(list)
    for trial_dir, usable_row_idx, split_dir_name in spotlight_specs:
        specs_by_trial[trial_dir].append((usable_row_idx, split_dir_name))

    for trial_dir, specs in tqdm(specs_by_trial.items(), disable=None):
        info = trial_info_by_dir[trial_dir]
        usable_frame_ids = info.usable_frame_ids
        if len(usable_frame_ids) == 0:
            raise ValueError(f"No usable frames for trial {trial_dir}")
        specs = sorted(specs, key=lambda x: x[0])
        requested_frame_indices = []
        for usable_idx, _split_dir_name in specs:
            if usable_idx >= len(usable_frame_ids):
                raise IndexError(
                    f"Usable frame selection index {usable_idx} out of range for {trial_dir.name}; "
                    f"only {len(usable_frame_ids)} usable frames available"
                )
            requested_frame_indices.append(int(usable_frame_ids[usable_idx]))

        # Read all requested frames from aligned behavior video in one shot.
        frames, _fps = read_frames_from_video(info.aligned_video_path, requested_frame_indices)
        frames_by_idx = {
            frame_idx: frame for frame_idx, frame in zip(requested_frame_indices, frames)
        }

        with h5py.File(info.prediction_h5_path, "r") as f:
            pred_xy = f["pred_xy"]
            pred_names = f.attrs["keypoint_names"].tolist()

            for usable_idx, split_dir_name in specs:
                frame_idx = int(usable_frame_ids[usable_idx])
                image_filename = f"frame_{frame_idx:09d}.jpg"
                if frame_idx not in frames_by_idx:
                    raise KeyError(
                        f"Frame {frame_idx} was requested from {info.aligned_video_path} "
                        "but was not returned by read_frames_from_video"
                    )
                frame = ensure_rgb(frames_by_idx[frame_idx])

                split_dir = output_dir / split_dir_name
                split_dir.mkdir(parents=True, exist_ok=True)
                output_filename = f"{trial_dir.name}_{image_filename}"
                output_image_path = split_dir / output_filename
                imageio.imwrite(output_image_path, frame)
                img_h, img_w = int(frame.shape[0]), int(frame.shape[1])

                if frame_idx >= pred_xy.shape[0]:
                    raise IndexError(
                        f"Frame index {frame_idx} out of range for {info.prediction_h5_path} "
                        f"with {pred_xy.shape[0]} frames"
                    )
                pred_xy_frame = pred_xy[frame_idx]

                xy_lookup = frame_xy_vis_from_prediction(
                    pred_xy_frame=pred_xy_frame,
                    pred_names=pred_names,
                )
                xy = build_endpoints(TARGET_SEGMENT_NAMES, xy_lookup, True)

                rel_path = str(output_image_path.relative_to(output_dir))
                sample = AnnotationSample(
                    rel_path=rel_path,
                    xy=xy,
                    image_hw=(img_h, img_w),
                    source_path=str(info.prediction_h5_path),
                )
                accumulators[split_dir_name].add(sample)


def main() -> None:
    # Define data paths.
    # Simulated images and keypoints (generated by NeuroMechFly processing).
    nmf_rendering_dir = Path("/Users/stimpfli/Desktop/prototype_data/bulk_data")
    # Spotlight trial root: each trial folder must contain
    # processed/aligned_behavior_video.mkv and poseforge_output/* files.
    spotlight_data_dir = Path("/Users/stimpfli/Desktop/prototype_data/poseforge/spotlight-data")
    # Output dataroot expected by CUT-style training.
    output_dir = Path("/Users/stimpfli/Desktop/prototype_data/bodypart_dataset")

    video_filename = "processed_nmf_sim_render_gray.mp4"

    # Sampling config.
    random_seed = 42
    np.random.seed(random_seed)
    n_train = 200
    n_test = 40
    n_val = 0

    # Define split mapping to CUT-style folder names.
    split_to_out = {"train": "train", "test": "test", "val": "val"}

    # Index available frames.
    nmf_num_frames = list_nmf_simulations_and_num_frames(nmf_rendering_dir, video_filename)
    spotlight_info = list_spotlight_recordings_and_usable_frames(
        spotlight_data_dir,
    )

    nmf_frame_pool = []
    for video_path, n_frames in nmf_num_frames.items():
        nmf_frame_pool.extend([(video_path, i) for i in range(n_frames)])

    spotlight_frame_pool = []
    for trial_dir, info in spotlight_info.items():
        spotlight_frame_pool.extend([(trial_dir, i) for i in range(len(info.usable_frame_ids))])

    total_requested = n_train + n_test + n_val
    if len(nmf_frame_pool) < total_requested:
        raise ValueError(
            f"Not enough synthetic frames: requested {total_requested}, available {len(nmf_frame_pool)}"
        )
    if len(spotlight_frame_pool) < total_requested:
        raise ValueError(
            f"Not enough spotlight usable frames: requested {total_requested}, "
            f"available {len(spotlight_frame_pool)}"
        )

    np.random.shuffle(nmf_frame_pool)
    np.random.shuffle(spotlight_frame_pool)

    selected: dict[str, dict[str, list[tuple[Path, int]]]] = {
        "train": {
            "nmf": nmf_frame_pool[:n_train],
            "spotlight": spotlight_frame_pool[:n_train],
        },
        "test": {
            "nmf": nmf_frame_pool[n_train : n_train + n_test],
            "spotlight": spotlight_frame_pool[n_train : n_train + n_test],
        },
    }
    if n_val > 0:
        val_start = n_train + n_test
        val_end = val_start + n_val
        selected["val"] = {
            "nmf": nmf_frame_pool[val_start:val_end],
            "spotlight": spotlight_frame_pool[val_start:val_end],
        }

    # Build extraction specs with output folder labels.
    nmf_specs: list[tuple[Path, int, str]] = []
    spotlight_specs: list[tuple[Path, int, str]] = []

    for split in selected.keys():
        split_prefix = split_to_out[split]
        for video_path, frame_idx in selected[split]["nmf"]:
            nmf_specs.append((video_path, frame_idx, f"{split_prefix}A"))
        for trial_dir, usable_row_idx in selected[split]["spotlight"]:
            spotlight_specs.append((trial_dir, usable_row_idx, f"{split_prefix}B"))

    # Prepare accumulators for NPZ export.
    accumulators = {
        "trainA": AnnotationAccumulator(split_name="train", domain_name="A", segment_names=TARGET_SEGMENT_NAMES),
        "trainB": AnnotationAccumulator(split_name="train", domain_name="B", segment_names=TARGET_SEGMENT_NAMES),
        "testA": AnnotationAccumulator(split_name="test", domain_name="A", segment_names=TARGET_SEGMENT_NAMES),
        "testB": AnnotationAccumulator(split_name="test", domain_name="B", segment_names=TARGET_SEGMENT_NAMES),
    }
    if n_val > 0:
        accumulators["valA"] = AnnotationAccumulator(
            split_name="val", domain_name="A", segment_names=TARGET_SEGMENT_NAMES
        )
        accumulators["valB"] = AnnotationAccumulator(
            split_name="val", domain_name="B", segment_names=TARGET_SEGMENT_NAMES
        )

    extract_synthetic_with_annotations(
        nmf_specs=nmf_specs,
        output_dir=output_dir,
        accumulators=accumulators,
    )
    extract_spotlight_with_annotations(
        spotlight_specs=spotlight_specs,
        output_dir=output_dir,
        trial_info_by_dir=spotlight_info,
        accumulators=accumulators,
    )

    annotations_dir = output_dir / "annotations"
    accumulators["trainA"].save(annotations_dir / "trainA_bodyparts.npz")
    accumulators["trainB"].save(annotations_dir / "trainB_bodyparts.npz")
    accumulators["testA"].save(annotations_dir / "testA_bodyparts.npz")
    accumulators["testB"].save(annotations_dir / "testB_bodyparts.npz")
    if n_val > 0:
        accumulators["valA"].save(annotations_dir / "valA_bodyparts.npz")
        accumulators["valB"].save(annotations_dir / "valB_bodyparts.npz")
    
    print(accumulators["trainA"].segment_names)


if __name__ == "__main__":
    main()