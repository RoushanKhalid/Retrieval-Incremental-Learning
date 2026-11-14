"""
Pytest configuration and shared fixtures for the Thesis automated test suite.
Uses synthetic datasets and isolated temporary directories to guarantee fast,
isolated, and repeatable execution without network access.
"""

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide an isolated temporary directory cleaned up after test completion."""
    d = Path(tempfile.mkdtemp(prefix="thesis_qa_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synthetic_image_dir(temp_dir: Path) -> Path:
    """Create a minimal synthetic cats vs dogs dataset directory with valid RGB images."""
    data_dir = temp_dir / "synthetic_data"
    cats_dir = data_dir / "cats"
    dogs_dir = data_dir / "dogs"
    cats_dir.mkdir(parents=True, exist_ok=True)
    dogs_dir.mkdir(parents=True, exist_ok=True)

    # Generate 10 cat images (predominantly reddish) and 10 dog images (predominantly bluish)
    rng = np.random.RandomState(42)
    for i in range(10):
        cat_arr = np.zeros((64, 64, 3), dtype=np.uint8)
        cat_arr[:, :, 0] = rng.randint(180, 255, (64, 64))
        cat_arr[:, :, 1] = rng.randint(0, 100, (64, 64))
        cat_arr[:, :, 2] = rng.randint(0, 100, (64, 64))
        Image.fromarray(cat_arr).save(cats_dir / f"cat_{i:04d}.jpg")

        dog_arr = np.zeros((64, 64, 3), dtype=np.uint8)
        dog_arr[:, :, 0] = rng.randint(0, 100, (64, 64))
        dog_arr[:, :, 1] = rng.randint(0, 100, (64, 64))
        dog_arr[:, :, 2] = rng.randint(180, 255, (64, 64))
        Image.fromarray(dog_arr).save(dogs_dir / f"dog_{i:04d}.jpg")

    return data_dir


class SyntheticTensorDataset(Dataset):
    """Simple tensor dataset for testing training loops without disk I/O."""

    def __init__(self, n_samples: int = 40, num_classes: int = 2, image_size: tuple[int, int] = (224, 224)):
        self.n_samples = n_samples
        self.data = torch.randn(n_samples, 3, *image_size)
        self.labels = torch.randint(0, num_classes, (n_samples,))
        self.paths = [f"synthetic_img_{i}.jpg" for i in range(n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        return self.data[idx], int(self.labels[idx]), self.paths[idx]


@pytest.fixture
def synthetic_dataloaders() -> dict[str, DataLoader]:
    """Provide minimal train, val, and test DataLoaders."""
    train_ds = SyntheticTensorDataset(32)
    val_ds = SyntheticTensorDataset(16)
    test_ds = SyntheticTensorDataset(16)
    return {
        "train": DataLoader(train_ds, batch_size=8, shuffle=True),
        "val": DataLoader(val_ds, batch_size=8, shuffle=False),
        "test": DataLoader(test_ds, batch_size=8, shuffle=False),
    }
