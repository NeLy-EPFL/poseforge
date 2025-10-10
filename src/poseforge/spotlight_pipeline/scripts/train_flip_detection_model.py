import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import json
from PIL import Image
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our custom modules
from poseforge.spotlight_pipeline.flip_detection_model import (
    create_model,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    run_inference,
)
from poseforge.spotlight_pipeline.flip_detection_dataset import (
    create_data_loaders,
    get_class_distribution,
    build_dataset_dataframe,
    get_transforms,
)
from poseforge.spotlight_pipeline.flip_detection_viz import (
    evaluate_model,
    save_training_plot,
)
from poseforge.util import set_random_seed


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    epoch: int,
    writer: SummaryWriter | None = None,
) -> tuple[float, float]:
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Log to tensorboard every 50 batches
        if writer and batch_idx % 50 == 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Loss/Train_Batch", loss.item(), step)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def train_model(
    dataframe: pd.DataFrame,
    output_dir: Path,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    image_size: int = 224,
    starting_checkpoint: Path | None = None,
    device: str = "auto",
    num_workers: int = 4,
    save_every: int = 10,
    seed: int = 42,
) -> None:
    # Set deterministic mode first
    set_random_seed(seed)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    print(f"Using dataset with {len(dataframe)} samples")

    # Print class distribution
    distribution, class_weights = get_class_distribution(dataframe)
    print("Class distribution:")
    for label, count in distribution.items():
        print(f"  {label}: {count}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataframe,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        seed=seed,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    model = create_model(num_classes=2, device=device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Calculate class weights for handling imbalanced data (move to device)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )

    # Load checkpoint if provided
    start_epoch = 0
    if starting_checkpoint:
        if starting_checkpoint.exists():
            start_epoch, _, _ = load_checkpoint(starting_checkpoint, model, optimizer)
            start_epoch += 1  # Start from next epoch

    # Setup tensorboard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = output_dir / f"runs/flip_detection_{timestamp}"
    writer = SummaryWriter(log_dir)

    # Save configuration and training dataset
    config = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "image_size": image_size,
        "starting_checkpoint": (
            str(starting_checkpoint) if starting_checkpoint else None
        ),
        "device": str(device),
        "num_workers": num_workers,
        "save_every": save_every,
        "seed": seed,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    dataframe.to_csv(output_dir / "dataset.csv", index=False)

    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Log to tensorboard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = output_dir / "best_model.pth"
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_model_path)
            print(f"New best validation accuracy: {best_val_acc:.2f}%")

        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)

    # Save final model
    final_model_path = output_dir / "final_model.pth"
    save_checkpoint(
        model,
        optimizer,
        num_epochs - 1,
        val_losses[-1],
        val_accs[-1],
        final_model_path,
    )

    # Save training curves
    plot_path = output_dir / "training_curves.png"
    save_training_plot(train_losses, val_losses, train_accs, val_accs, plot_path)

    # Final evaluation on test set
    if len(test_loader) > 0:
        print("\nEvaluating on test set...")
        test_preds, test_labels, test_acc, test_report = evaluate_model(
            model, test_loader, device
        )
        print(f"Test Accuracy: {test_acc:.4f}")

        # Save test results
        test_results = {"accuracy": test_acc, "classification_report": test_report}
        with open(output_dir / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)

    writer.close()
    print(f"\nTraining completed! Results saved to: {output_dir}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"TensorBoard logs: {log_dir}")


def run_inference_on_test_dataset(
    test_dataframe: pd.DataFrame,
    model: nn.Module,
    output_dir: Path,
    image_size: int = 224,
    device: torch.device | str = "cpu",
) -> None:
    # Make explicit copy to avoid pandas warnings
    test_df = test_dataframe.copy()

    # Load model and make inference
    print("Running inference on test set...")
    transform = get_transforms("test", image_size)
    model = model.to(device)
    labels = []
    confidences = []
    for idx, row in tqdm(
        test_df.iterrows(),
        total=len(test_df),
        disable=None,
        desc="Inference",
    ):
        image_path = Path(row["path"])
        if not image_path.exists():
            print(f"Image {image_path} does not exist, skipping...")
            continue

        # Load and preprocess image
        image = Image.open(image_path).convert("L")
        label, conf = run_inference(
            model,
            image,
            transform=transform,
            device=device,
            labels=["not flipped", "flipped"],
        )
        labels.append(label)
        confidences.append(conf)

    test_df["predicted_label"] = labels
    test_df["predicted_confidence"] = confidences

    # Save test example images
    print("Copying test examples to appropriate directories for inspection...")
    test_examples_dir = output_dir / "test_examples"
    for true_label in ["not flipped", "flipped"]:
        for pred_label in ["not flipped", "flipped"]:
            true_label_ = true_label.replace(" ", "_")
            pred_label_ = pred_label.replace(" ", "_")
            dir_ = test_examples_dir / f"{true_label_}_predicted_as_{pred_label_}"
            dir_.mkdir(parents=True, exist_ok=True)
            sel = test_df[
                (test_df["label"] == true_label)
                & (test_df["predicted_label"] == pred_label)
            ]
            for idx, row in sel.iterrows():
                image_path = Path(row["path"])
                if not image_path.exists():
                    print(f"Image {image_path} does not exist, skipping...")
                    continue
                confidence = row["predicted_confidence"]
                image = Image.open(image_path).convert("L")
                image.save(dir_ / f"{idx:06d}_confidence_{confidence:.2f}.jpg")
    test_df.to_csv(test_examples_dir / "test_results.csv", index=False)


if __name__ == "__main__":
    retrain_model = False

    # Load dataset
    print("Loading dataset...")
    image_base_dir = Path("bulk_data/behavior_images/spotlight_aligned_and_cropped")
    train_trials = [
        "20250613-fly1b-001",
        "20250613-fly1b-002",
        "20250613-fly1b-003",
        "20250613-fly1b-004",
        "20250613-fly1b-005",
        "20250613-fly1b-006",
        "20250613-fly1b-007",
        "20250613-fly1b-008",
        "20250613-fly1b-009",
        "20250613-fly1b-010",
        "20250613-fly1b-011",
        "20250613-fly1b-013",
        "20250613-fly1b-014",
    ]
    val_trials = ["20250613-fly1b-012"]
    test_trials = ["20250613-fly1b-015"]
    dataframe = build_dataset_dataframe(
        train_trials_dirs=[image_base_dir / trial for trial in train_trials],
        val_trials_dirs=[image_base_dir / trial for trial in val_trials],
        test_trials_dirs=[image_base_dir / trial for trial in test_trials],
    )
    print(f"Loaded dataset with {len(dataframe)} samples")

    # Train model
    output_dir = Path("bulk_data/flip_detection_model")
    if retrain_model:
        train_model(
            dataframe,
            output_dir,
            num_epochs=50,
            batch_size=32,
            learning_rate=0.001,
            image_size=224,
            starting_checkpoint=None,
            device="auto",
            num_workers=4,
            save_every=10,
            seed=42,
        )

    # Run inference on test dataset
    model = create_model(num_classes=2)
    model_path = output_dir / "best_model.pth"
    epoch, loss, accuracy = load_checkpoint(model_path, model)
    print(
        f"Loaded model from {model_path} at epoch {epoch}: "
        f"loss: {loss:.4f}, accuracy: {accuracy:.2f}%"
    )
    test_dataframe = dataframe[dataframe["dataset"] == "test"]
    run_inference_on_test_dataset(
        test_dataframe, model, output_dir, image_size=224, device="cpu"
    )
