import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

from biomechpose.spotlight_pipeline.flip_detection_dataset import get_transforms


class FlipDetectionCNN(nn.Module):
    """Simple CNN for binary classification (flipped vs not flipped)"""

    def __init__(self, input_channels: int = 1, num_classes: int = 2):
        super(FlipDetectionCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Adaptive pooling
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def create_model(
    num_classes: int = 2, device: str | torch.device = "cuda"
) -> nn.Module:
    """Create and return the model"""
    model = FlipDetectionCNN(input_channels=1, num_classes=num_classes)
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    checkpoint_path: Path,
) -> None:
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "accuracy": accuracy,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, float, float]:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))
    accuracy = checkpoint.get("accuracy", 0.0)

    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Resuming from epoch {epoch}, loss: {loss:.4f}, accuracy: {accuracy:.2f}%")

    return epoch, loss, accuracy


def run_inference(
    model: nn.Module,
    image: np.ndarray | Image.Image,
    transform: transforms.Compose,
    device: torch.device,
    labels: list[str] = ["not flipped", "flipped"],
) -> tuple[str, float]:
    """
    Run inference on a single image and return the predicted label and confidence.

    Args:
        model: Trained PyTorch model.
        image: Input image as a numpy array (num_rows, num_columns).
        transform: Transform function to preprocess the image.
        device: Device to run inference on.

    Returns:
        Tuple of (predicted_label, confidence)
    """
    model.eval()
    with torch.no_grad():
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        # Apply transform
        input_tensor = transform(image)
        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0).to(device)
        # Forward pass
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
        label = labels[pred.item()]
        return label, conf.item()
