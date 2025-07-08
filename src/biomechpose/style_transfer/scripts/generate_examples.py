import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
import torch
import numpy as np
from typing import List
import random
from torchvision import transforms
from PIL import Image

from biomechpose.style_transfer.model import CycleGAN
from biomechpose.style_transfer.utils import load_model_for_inference, get_device, set_all_seeds
from biomechpose.style_transfer.visualize import tensor_to_numpy

@dataclass
class GenerateExamplesConfig:
    model_path: Path  # Path to trained model (.pth file)
    output_dir: Path | None = None
    test_experimental_images_dir: Path = Path(
        "bulk_data/style_transfer/kinematic_recording/spotlight202506/spotlight_recordings_unified/test"
    )
    test_simulated_images_dir: Path = Path(
        "bulk_data/style_transfer/simulated_frames/aymanns2022_per_frame/test"
    )
    random_seed: int = 42
    image_size: int = 512  # Default, will be overridden by model if available
    device: str | None = None
    n_examples: int = 10

def get_image_paths(image_dir: Path) -> List[Path]:
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(image_dir.glob(f"*{ext}"))
        image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
    return sorted(image_paths)

def load_and_preprocess_image(image_path: Path, direction: str, image_size: int) -> torch.Tensor:
    image = Image.open(image_path)
    if direction == "sim_to_exp":
        image = image.convert("RGB")
        normalize_mean = [0.5, 0.5, 0.5]
        normalize_std = [0.5, 0.5, 0.5]
    else:
        image = image.convert("L")
        normalize_mean = [0.5]
        normalize_std = [0.5]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor

def plot_examples_grid(rows: List[List[np.ndarray]], titles: List[str], save_path: Path):
    n_rows = len(rows)
    n_cols = len(rows[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*2.5))
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            img = rows[i][j]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.axis("off")
            if i == 0:
                ax.set_title(titles[j])
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def generate_examples(config: GenerateExamplesConfig):
    if config.output_dir is None:
        output_dir = config.model_path.parent.parent / "generated_examples"
    else:
        output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # Set random seed
    set_all_seeds(config.random_seed)
    # Device
    device = get_device() if config.device is None else torch.device(config.device)
    # --- Load checkpoint hyperparams first ---
    checkpoint = torch.load(config.model_path, map_location=device, weights_only=False)
    hyperparams = checkpoint.get("hyperparameters", {})
    # Instantiate model with correct architecture
    model = CycleGAN(
        generator_base_filters=hyperparams.get("generator_base_filters", 64),
        generator_n_residual_blocks=hyperparams.get("generator_n_residual_blocks", 6),
        generator_use_dropout=hyperparams.get("generator_use_dropout", False),
        discriminator_base_filters=hyperparams.get("discriminator_base_filters", 64),
        discriminator_n_layers=hyperparams.get("discriminator_n_layers", 3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    image_size = hyperparams.get("image_size", config.image_size)
    # --- Sim2Exp2Sim ---
    sim_paths = get_image_paths(config.test_simulated_images_dir)
    if len(sim_paths) < config.n_examples:
        raise RuntimeError(f"Not enough simulated images in {config.test_simulated_images_dir}")
    sim_selected = random.sample(sim_paths, config.n_examples)
    sim2exp2sim_rows = []
    for sim_path in sim_selected:
        # Simulated image
        sim_tensor = load_and_preprocess_image(sim_path, "sim_to_exp", image_size).to(device)
        # Sim -> Exp
        with torch.no_grad():
            fake_exp = model.G_sim_to_exp(sim_tensor)
            # Exp -> Sim
            fake_sim = model.G_exp_to_sim(fake_exp)
        # Convert to numpy for plotting
        sim_np = tensor_to_numpy(sim_tensor.cpu())
        fake_exp_np = tensor_to_numpy(fake_exp.cpu())
        fake_sim_np = tensor_to_numpy(fake_sim.cpu())
        sim2exp2sim_rows.append([sim_np, fake_exp_np, fake_sim_np])
    plot_examples_grid(
        sim2exp2sim_rows,
        ["Simulated", "Sim→Exp (pred)", "Sim→Exp→Sim (pred)"],
        output_dir / f"{config.model_path.name}_example_sim2exp2sim.png"
    )
    # --- Exp2Sim2Exp ---
    exp_paths = get_image_paths(config.test_experimental_images_dir)
    if len(exp_paths) < config.n_examples:
        raise RuntimeError(f"Not enough experimental images in {config.test_experimental_images_dir}")
    exp_selected = random.sample(exp_paths, config.n_examples)
    exp2sim2exp_rows = []
    for exp_path in exp_selected:
        # Experimental image
        exp_tensor = load_and_preprocess_image(exp_path, "exp_to_sim", image_size).to(device)
        # Exp -> Sim
        with torch.no_grad():
            fake_sim = model.G_exp_to_sim(exp_tensor)
            # Sim -> Exp
            fake_exp = model.G_sim_to_exp(fake_sim)
        # Convert to numpy for plotting
        exp_np = tensor_to_numpy(exp_tensor.cpu())
        fake_sim_np = tensor_to_numpy(fake_sim.cpu())
        fake_exp_np = tensor_to_numpy(fake_exp.cpu())
        exp2sim2exp_rows.append([exp_np, fake_sim_np, fake_exp_np])
    plot_examples_grid(
        exp2sim2exp_rows,
        ["Experimental", "Exp→Sim (pred)", "Exp→Sim→Exp (pred)"],
        output_dir / f"{config.model_path.name}_example_exp2sim2exp.png"
    )

def main():
    import tyro
    config = tyro.cli(GenerateExamplesConfig)
    generate_examples(config)

if __name__ == "__main__":
    main()