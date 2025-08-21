import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path


class FlipDetectionDataset(Dataset):
    """Custom dataset for flip detection"""

    def __init__(
        self, dataframe: pd.DataFrame, transform: transforms.Compose | None = None
    ):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

        # Create label mapping
        self.label_to_idx = {"not flipped": 0, "flipped": 1}
        self.idx_to_label = {0: "not flipped", 1: "flipped"}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # Load image
        image_path = row["path"]
        try:
            image = Image.open(image_path).convert("L")  # Convert to grayscale
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

        # Get label
        label_str = row["label"]
        if label_str not in self.label_to_idx:
            raise ValueError(f"Unknown label: {label_str}")
        label = self.label_to_idx[label_str]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(mode: str, image_size: int) -> transforms.Compose:
    """Get transforms for different modes"""

    if mode == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
            ]
        )
    elif mode in ["val", "test"]:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'train', 'val', or 'test'.")


def create_data_loaders(
    df: pd.DataFrame,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, validation, and test sets"""

    # Split dataframe by dataset type
    train_df = df[df["dataset"] == "train"].copy()
    val_df = df[df["dataset"] == "val"].copy()
    test_df = df[df["dataset"] == "test"].copy()

    # Create transforms
    train_transform = get_transforms("train", image_size)
    val_transform = get_transforms("val", image_size)
    test_transform = get_transforms("test", image_size)

    # Create datasets
    train_dataset = FlipDetectionDataset(train_df, train_transform)
    val_dataset = FlipDetectionDataset(val_df, val_transform)
    test_dataset = FlipDetectionDataset(test_df, test_transform)

    # Create generator for deterministic data loading
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
    )

    return train_loader, val_loader, test_loader


def get_class_distribution(
    train_dataframe: pd.DataFrame, labels: list[str] = ["not flipped", "flipped"]
) -> tuple[dict[str, int], torch.Tensor]:
    """
    Calculates the distribution of classes in a training DataFrame and
    computes class weights for imbalanced datasets.

    Args:
        train_dataframe (pd.DataFrame): The DataFrame containing the
            training data with a 'label' column.
        labels (list[str], optional): List of class labels to consider.

    Returns:
        - A dictionary mapping each class label to its count in the
          DataFrame.
        - A tensor containing the computed class weights for each label,
          suitable for use in loss functions.
    """
    class_counts = train_dataframe["label"].value_counts().to_dict()
    weight_tensor = torch.tensor(
        [len(train_dataframe) / (2 * class_counts.get(label, 1)) for label in labels],
        dtype=torch.float32,
    )
    return class_counts, weight_tensor


def build_dataset_dataframe(
    train_trials_dirs: list[Path],
    val_trials_dirs: list[Path],
    test_trials_dirs: list[Path],
) -> pd.DataFrame:
    dataframes = []
    for dataset, dirs in zip(
        ["train", "val", "test"], [train_trials_dirs, val_trials_dirs, test_trials_dirs]
    ):
        image_labels = []
        for trial_dir in dirs:
            for image_path in sorted(trial_dir.glob(f"not_flipped/*.jpg")):
                image_labels.append((dataset, "not flipped", image_path.absolute()))
            for image_path in sorted(trial_dir.glob(f"flipped/*.jpg")):
                image_labels.append((dataset, "flipped", image_path.absolute()))
        dataframe_per_dataset = pd.DataFrame(
            image_labels, columns=["dataset", "label", "path"]
        )
        dataframes.append(dataframe_per_dataset)
    return pd.concat(dataframes, ignore_index=True)
