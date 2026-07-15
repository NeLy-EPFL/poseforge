import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
from importlib.resources import files
import yaml

from poseforge.spotlight.flip_detection.model import (
    create_model,
    load_checkpoint,
)
from poseforge.spotlight.flip_detection.dataset import get_transforms

LABELS = ["not flipped", "flipped"]


class _InferenceImageDataset(Dataset):
    """Loads grayscale images from a list of paths for batched inference."""

    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("L")
        return self.transform(image), idx


def run_batched_inference(model, image_paths, transform, device, batch_size, n_workers):
    """Run the flip detector over all images and return labels in input order."""
    dataset = _InferenceImageDataset(image_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=(device == "cuda"),
    )

    labels = [None] * len(image_paths)
    model.eval()
    use_amp = device == "cuda"
    with torch.no_grad():
        for images, idxs in tqdm(loader, desc="Detecting flips"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            for pred, idx in zip(preds, idxs.tolist()):
                labels[idx] = LABELS[pred]
    return labels


def start():
    parser = argparse.ArgumentParser(
        description="Detect flipped flies in spotlight recordings."
    )
    parser.add_argument(
        "aligned_data_dir",
        type=Path,
        default=Path("bulk_data/spotlight_aligned_and_cropped"),
        help="Base directory containing aligned and cropped spotlight recording trials.",
    )
    parser.add_argument(
        "glob_pattern",
        type=str,
        default="fly*",
        help="Glob pattern to match spotlight trial directories.",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=files("poseforge").joinpath("production/spotlight/config.yaml"),
    )
    parser.add_argument(
        "--detection_model_dir",
        type=Path,
        help="Path to flip detection model directory. If not provided, will be loaded from config file.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on: 'auto' (default), 'cuda', or 'cpu'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Inference batch size. Defaults to config flip_detection.batch_size or 256.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="DataLoader workers. Defaults to config common.n_workers or 8.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = start()

    # Load config (needed for checkpoint path and default batch/worker counts).
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    flip_detection_model_dir = args.detection_model_dir
    if not flip_detection_model_dir:
        flip_detection_model_dir = Path(config["flip_detection"]["checkpoint"]).parent

    # Resolve device.
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but no CUDA device is available.")
    print(f"Running flip detection on device: {device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Resolve batch size / workers (CLI > config > default).
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = config.get("flip_detection", {}).get("batch_size", 256)
    n_workers = args.n_workers
    if n_workers is None:
        n_workers = config.get("common", {}).get(
            "n_workers", config.get("flip_detection", {}).get("n_workers", 8)
        )

    image_size = 224  # flip detector runs at this resolution
    link_data = True

    # Load model and transforms.
    model = create_model(num_classes=2, device=device)
    checkpoint_path = flip_detection_model_dir / "best_model.pth"
    _, _, accuracy = load_checkpoint(checkpoint_path, model)
    print(f"Loaded model from {checkpoint_path} with accuracy: {accuracy:.2f}%")
    transforms = get_transforms("test", image_size=image_size)

    print(f"Using batch size {batch_size} with {n_workers} dataloader workers.")

    # Process each trial.
    for trial in sorted(args.aligned_data_dir.glob(args.glob_pattern)):
        if not trial.is_dir():
            continue
        print(f"Processing trial: {trial.name}")

        all_images = sorted(list(trial.glob("all/*.jpg")))
        if not all_images:
            print(f"  No images found in {trial / 'all'}, skipping.")
            continue

        labels_all = run_batched_inference(
            model, all_images, transforms, device, batch_size, n_workers
        )
        df = pd.DataFrame(
            {"image": [x.name for x in all_images], "predicted_label": labels_all}
        )
        df.to_csv(trial / "predicted_flip_labels.csv", index=False)

        if link_data:
            # Make one folder per label and symlink images into it. Downstream
            # steps (bodyseg, keypoints3d) read from model_prediction/not_flipped.
            label_dirs = {}
            for label in LABELS:
                label_dir = trial / "model_prediction" / label.replace(" ", "_")
                label_dir.mkdir(exist_ok=True, parents=True)
                label_dirs[label] = label_dir
            for name, label in zip(df["image"], df["predicted_label"]):
                src = trial / "all" / name
                dst = label_dirs[label] / name
                if not dst.exists():
                    dst.symlink_to(src.resolve())
