from email import parser
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
from importlib.resources import files
import yaml

from poseforge.spotlight.flip_detection.model import (
    create_model,
    load_checkpoint,
    run_inference,
)
from poseforge.spotlight.flip_detection.dataset import get_transforms

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
    # get package root path for default config path
    parser.add_argument(
        "--config_path",
        type=Path,
        # path relative to poseforge package root
        default=files("poseforge").joinpath(
            "production/spotlight/config.yaml"
        ),
    )
    # make optional
    parser.add_argument(
        "--detection_model_dir",
        type=Path,
        help="Path to flip detection model directory. If not provided, will be loaded from config file.",
        required=False,
        default=None,
    )
    args = parser.parse_args()

    return args.aligned_data_dir, args.glob_pattern, args.config_path, args.detection_model_dir


if __name__ == "__main__":
    # parse paths
    spotlight_data_dir, glob_pattern, config_path, flip_detection_model_dir = start()
    if not flip_detection_model_dir:
        # load from config file
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        flip_detection_model_dir = Path(config["flip_detection"]["checkpoint"]).parent

    image_size = 224  # flip detector runs at this resolution
    device = "cpu"  # Use CPU for inference
    link_data = True

    # Load model and transforms
    model = create_model(num_classes=2, device=device)
    checkpoint_path = flip_detection_model_dir / "best_model.pth"
    _, _, accuracy = load_checkpoint(checkpoint_path, model)
    print(f"Loaded model from {checkpoint_path} with accuracy: {accuracy:.2f}%")
    transforms = get_transforms("test", image_size=image_size)

    # Process each trial
    for trial in spotlight_data_dir.glob(glob_pattern):
        if not trial.is_dir():
            continue
        print(f"Processing trial: {trial.name}")
        # Run inference on all images in the trial
        all_images = sorted(list(trial.glob("all/*.jpg")))
        labels_all = []
        for image_path in tqdm(all_images):
            image = Image.open(image_path).convert("L")
            label, confidence = run_inference(model, image, transforms, device=device)
            labels_all.append(label)
        df = pd.DataFrame(
            {"image": [x.name for x in all_images], "predicted_label": labels_all}
        )
        df.to_csv(trial / "predicted_flip_labels.csv", index=False)

        if link_data:
            # Make new folders for each label and link images to those folders
            # This helps manual inspection of predictions
            for label in ["not flipped", "flipped"]:
                label_dir = trial / "model_prediction" / label.replace(" ", "_")
                label_dir.mkdir(exist_ok=True, parents=True)
                for i, row in df.iterrows():
                    if row["predicted_label"] == label:
                        src = trial / "all" / row["image"]
                        dst = label_dir / row["image"]
                        if not dst.exists():
                            dst.symlink_to(src.resolve())
