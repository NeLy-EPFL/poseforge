import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torch.utils.data import DataLoader
from pathlib import Path


def show_sample_images(
    dataset: torch.utils.data.Dataset,
    num_samples: int = 8,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """Show sample images from the dataset"""
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.ravel()

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, idx in enumerate(indices):
        image, label = dataset[idx]

        # Convert tensor to numpy and denormalize
        if isinstance(image, torch.Tensor):
            img_np = image.squeeze().numpy()
            img_np = (img_np + 1) / 2  # Denormalize from [-1,1] to [0,1]
        else:
            img_np = np.array(image)

        axes[i].imshow(img_np, cmap="gray")
        axes[i].set_title(f"Label: {dataset.idx_to_label[label.item()]}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | str,
    class_names: list[str] = ["not flipped", "flipped"],
) -> tuple[list[int], list[int], float, dict[str, dict[str, float]]]:
    """Evaluate model and return predictions, true labels, and metrics"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    # Generate classification report
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )

    return all_preds, all_labels, accuracy, report


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] = ["not flipped", "flipped"],
    figsize: tuple[int, int] = (6, 5),
) -> np.ndarray:
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    return cm


def show_predictions(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device | str,
    num_samples: int = 8,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """Show model predictions on sample images"""
    model.eval()

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.ravel()

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, true_label = dataset[idx]

            # Get prediction
            image_batch = image.unsqueeze(0).to(device)
            output = model(image_batch)
            _, predicted = torch.max(output, 1)
            confidence = torch.softmax(output, dim=1).max().item()

            # Convert tensor to numpy and denormalize
            img_np = image.squeeze().numpy()
            img_np = (img_np + 1) / 2  # Denormalize from [-1,1] to [0,1]

            # Plot
            axes[i].imshow(img_np, cmap="gray")

            true_label_str = dataset.idx_to_label[true_label.item()]
            pred_label_str = dataset.idx_to_label[predicted.item()]

            color = "green" if true_label == predicted else "red"
            axes[i].set_title(
                f"True: {true_label_str}\nPred: {pred_label_str}\nConf: {confidence:.2f}",
                color=color,
            )
            axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def print_model_summary(
    model: nn.Module, input_size: tuple[int, int, int] = (1, 224, 224)
) -> None:
    """Print a summary of the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(f"Input size: {input_size}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 50)

    # Try to print model structure
    try:
        dummy_input = torch.randn(1, *input_size)
        print("\nModel structure:")
        print(model)
    except:
        print("Could not display model structure")


def save_training_plot(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    save_path: Path,
) -> None:
    """Save training curves to file"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    # Plot losses
    ax1.plot(epochs, train_losses, "b-", label="Training Loss")
    ax1.plot(epochs, val_losses, "r-", label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(epochs, train_accs, "b-", label="Training Accuracy")
    ax2.plot(epochs, val_accs, "r-", label="Validation Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {save_path}")
