"""
Inference script for CycleGAN fruit fly image translation.

Example usage:
    # Translate simulated to experimental
    python infer.py --model_path ./outputs/models/best_model.pth --input_image ./test_sim.png --direction sim_to_exp --output_rundir ./results

    # Translate experimental to simulated
    python infer.py --model_path ./outputs/models/best_model.pth --input_image ./test_exp.png --direction exp_to_sim --output_rundir ./results

    # Batch inference on directory
    python infer.py --model_path ./outputs/models/best_model.pth --input_dir ./test_images --direction sim_to_exp --output_rundir ./results
"""

import tyro
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, List

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from biomechpose.style_transfer.model import CycleGAN
from biomechpose.style_transfer.utils import load_model_for_inference, get_device
from biomechpose.style_transfer.visualize import (
    create_inference_visualization,
    tensor_to_numpy,
)


@dataclass
class InferenceConfig:
    """Inference configuration."""

    # === Required Parameters ===
    model_path: Path  # Path to trained model (.pth file)
    direction: Literal["sim_to_exp", "exp_to_sim"]  # Translation direction
    output_rundir: Path  # Output directory for results

    # === Input Configuration ===
    input_image: Optional[Path] = (
        None  # Single input image (mutually exclusive with input_dir)
    )
    input_dir: Optional[Path] = (
        None  # Directory of input images (mutually exclusive with input_image)
    )

    # === Model Configuration ===
    device: Optional[str] = (
        None  # Device to use ('cuda', 'cpu', 'mps', or None for auto)
    )

    # === Output Configuration ===
    save_visualization: bool = True  # Save side-by-side visualization
    save_generated_only: bool = True  # Save only the generated image
    output_format: Literal["png", "jpg"] = "png"  # Output image format

    # === Processing Configuration ===
    image_size: Optional[int] = (
        None  # Resize images (None to use model's training size)
    )


def load_and_preprocess_image(
    image_path: Path, direction: str, image_size: int = 900
) -> torch.Tensor:
    """Load and preprocess an image for inference."""

    # Load image
    image = Image.open(image_path)

    # Convert based on direction
    if direction == "sim_to_exp":
        # Input should be RGB (simulated)
        image = image.convert("RGB")
        normalize_mean = [0.5, 0.5, 0.5]
        normalize_std = [0.5, 0.5, 0.5]
    else:  # exp_to_sim
        # Input should be grayscale (experimental)
        image = image.convert("L")
        normalize_mean = [0.5]
        normalize_std = [0.5]

    # Create transform
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ]
    )

    # Apply transform and add batch dimension
    tensor = transform(image).unsqueeze(0)

    return tensor


def postprocess_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Convert model output tensor to numpy array for saving."""
    # Remove batch dimension and move to CPU
    tensor = tensor.squeeze(0).cpu()

    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy
    if tensor.shape[0] == 1:  # Grayscale
        return tensor.squeeze(0).numpy()
    elif tensor.shape[0] == 3:  # RGB
        return tensor.permute(1, 2, 0).numpy()
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")


def save_image_array(array: np.ndarray, save_path: Path, format: str = "png"):
    """Save numpy array as image."""
    if len(array.shape) == 2:  # Grayscale
        # Convert to 8-bit
        array_8bit = (array * 255).astype(np.uint8)
        image = Image.fromarray(array_8bit, mode="L")
    else:  # RGB
        # Convert to 8-bit
        array_8bit = (array * 255).astype(np.uint8)
        image = Image.fromarray(array_8bit, mode="RGB")

    # Ensure directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save image
    if format.lower() == "jpg":
        image.save(save_path, format="JPEG", quality=95)
    else:
        image.save(save_path, format="PNG")


def get_image_paths(image_dir: Path) -> List[Path]:
    """Get all image paths from directory."""
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    image_paths = []

    for ext in extensions:
        image_paths.extend(image_dir.glob(f"*{ext}"))
        image_paths.extend(image_dir.glob(f"*{ext.upper()}"))

    return sorted(image_paths)


def infer_single_image(
    model: CycleGAN, image_path: Path, config: InferenceConfig, device: torch.device
):
    """Run inference on a single image."""

    print(f"Processing: {image_path}")

    # Load and preprocess image
    image_size = config.image_size if config.image_size is not None else 900
    input_tensor = load_and_preprocess_image(image_path, config.direction, image_size)
    input_tensor = input_tensor.to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        if config.direction == "sim_to_exp":
            generated_tensor = model.G_sim_to_exp(input_tensor)
        else:  # exp_to_sim
            generated_tensor = model.G_exp_to_sim(input_tensor)

    # Create output filename
    stem = image_path.stem
    output_stem = f"{stem}_{config.direction}"

    # Save generated image only
    if config.save_generated_only:
        generated_array = postprocess_tensor(generated_tensor)
        generated_path = config.output_rundir / f"{output_stem}.{config.output_format}"
        save_image_array(generated_array, generated_path, config.output_format)
        print(f"Saved generated image: {generated_path}")

    # Save visualization (side-by-side)
    if config.save_visualization:
        vis_path = config.output_rundir / f"{output_stem}_comparison.png"
        create_inference_visualization(
            input_tensor, generated_tensor, config.direction, vis_path
        )
        print(f"Saved visualization: {vis_path}")


def main(config: InferenceConfig):
    """Main inference function."""

    # Validate input arguments
    if config.input_image is None and config.input_dir is None:
        raise ValueError("Either input_image or input_dir must be specified")

    if config.input_image is not None and config.input_dir is not None:
        raise ValueError("Cannot specify both input_image and input_dir")

    # Setup device
    if config.device is None:
        device = get_device()
    else:
        device = torch.device(config.device)

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {config.model_path}")

    # Create model with default parameters (will be overridden by checkpoint)
    model = CycleGAN()

    # Load trained weights and hyperparameters
    hyperparams = load_model_for_inference(config.model_path, model, device)

    print("Model loaded successfully")
    if hyperparams:
        print(
            f"Model was trained with image size: {hyperparams.get('image_size', 'unknown')}"
        )
        if config.image_size is None:
            config.image_size = hyperparams.get("image_size", 900)

    # Create output directory
    config.output_rundir.mkdir(parents=True, exist_ok=True)

    # Process images
    if config.input_image is not None:
        # Single image inference
        if not config.input_image.exists():
            raise FileNotFoundError(f"Input image not found: {config.input_image}")

        infer_single_image(model, config.input_image, config, device)

    else:
        # Batch inference
        if not config.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {config.input_dir}")

        image_paths = get_image_paths(config.input_dir)

        if len(image_paths) == 0:
            print(f"No images found in: {config.input_dir}")
            return

        print(f"Found {len(image_paths)} images for processing")

        for i, image_path in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] ", end="")
            try:
                infer_single_image(model, image_path, config, device)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

    print("Inference completed!")


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)
