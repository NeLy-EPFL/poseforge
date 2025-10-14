import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from poseforge.spotlight.flip_detection.model import (
    create_model,
    load_checkpoint,
    run_inference,
)
from poseforge.spotlight.flip_detection.dataset import get_transforms


if __name__ == "__main__":
    # Define paths and parameters
    flip_detection_model_dir = Path("bulk_data/flip_detection_model")
    spotlight_data_dir = Path("bulk_data/behavior_images/spotlight_aligned_and_cropped")
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
    for trial in sorted(spotlight_data_dir.iterdir()):
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
