"""
Dataset loader and partitioning module for the Thesis benchmarking framework.
Provides deterministic 3-way stratified split: Train (70%), Val (15%), Test (15%).
"""

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CatsDogsDataset(Dataset):
    """PyTorch Dataset for Cat vs Dog image classification."""

    def __init__(self, file_paths: list[Path], labels: list[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        path = self.file_paths[idx]
        label = self.labels[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                tensor = self.transform(img)
            else:
                tensor = transforms.ToTensor()(img)
        return tensor, label, str(path)


def get_default_transforms(image_size: tuple[int, int] = (224, 224)):
    """Return train and eval transforms."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform


def prepare_stratified_split(
    data_dir: Path,
    seed: int = 777,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    cache_file: Path | None = None
) -> dict[str, tuple[list[Path], list[int]]]:
    """
    Produce a deterministic 3-way stratified partition: train, validation, and test.
    Ensures identical sample distribution across all downstream models.
    """
    if cache_file and cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        return {
            split: (
                [Path(p) for p in data["paths"]],
                data["labels"]
            )
            for split, data in cached_data.items()
        }

    classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    splits: dict[str, dict[str, list]] = {
        "train": {"paths": [], "labels": []},
        "val": {"paths": [], "labels": []},
        "test": {"paths": [], "labels": []},
    }

    rng = random.Random(seed)

    for class_idx, class_name in enumerate(classes):
        class_dir = data_dir / class_name
        files = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png")))
        rng.shuffle(files)

        n_total = len(files)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)
        n_train = n_total - n_val - n_test

        val_files = files[:n_val]
        test_files = files[n_val : n_val + n_test]
        train_files = files[n_val + n_test :]

        for f in train_files:
            splits["train"]["paths"].append(str(f))
            splits["train"]["labels"].append(class_idx)

        for f in val_files:
            splits["val"]["paths"].append(str(f))
            splits["val"]["labels"].append(class_idx)

        for f in test_files:
            splits["test"]["paths"].append(str(f))
            splits["test"]["labels"].append(class_idx)

    # Shuffle training partition deterministically
    train_indices = list(range(len(splits["train"]["paths"])))
    rng.shuffle(train_indices)
    splits["train"]["paths"] = [splits["train"]["paths"][i] for i in train_indices]
    splits["train"]["labels"] = [splits["train"]["labels"][i] for i in train_indices]

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)

    return {
        split: (
            [Path(p) for p in data["paths"]],
            data["labels"]
        )
        for split, data in splits.items()
    }


def get_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    seed: int = 777,
    image_size: tuple[int, int] = (224, 224),
    num_workers: int = 0,
    cache_file: Path | None = None
) -> tuple[dict[str, DataLoader], dict[str, tuple[list[Path], list[int]]]]:
    """Build PyTorch DataLoaders for train, val, and test splits."""
    splits = prepare_stratified_split(
        data_dir=data_dir,
        seed=seed,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        cache_file=cache_file
    )

    train_tf, eval_tf = get_default_transforms(image_size)

    train_ds = CatsDogsDataset(splits["train"][0], splits["train"][1], transform=train_tf)
    val_ds = CatsDogsDataset(splits["val"][0], splits["val"][1], transform=eval_tf)
    test_ds = CatsDogsDataset(splits["test"][0], splits["test"][1], transform=eval_tf)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    return loaders, splits
