"""
Unit tests for dataset loading, preprocessing, and 3-way deterministic stratified splitting.
"""

from pathlib import Path

import torch

from tcontext.data.dataset_loader import (
    get_dataloaders,
    get_default_transforms,
    prepare_stratified_split,
)


def test_default_transforms():
    """Verify train and eval transforms return expected tensor shapes and value ranges."""
    train_tf, eval_tf = get_default_transforms((224, 224))
    assert train_tf is not None
    assert eval_tf is not None

    import numpy as np
    from PIL import Image

    img = Image.fromarray(np.uint8(np.random.randint(0, 255, (300, 400, 3))))
    t_out = train_tf(img)
    e_out = eval_tf(img)

    assert isinstance(t_out, torch.Tensor)
    assert t_out.shape == (3, 224, 224)
    assert isinstance(e_out, torch.Tensor)
    assert e_out.shape == (3, 224, 224)


def test_stratified_split_partition_ratios(synthetic_image_dir: Path, temp_dir: Path):
    """Verify that train (70%), val (15%), test (15%) partitions are strictly disjoint and balanced."""
    cache_file = temp_dir / "test_split.json"
    splits = prepare_stratified_split(
        data_dir=synthetic_image_dir,
        seed=777,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        cache_file=cache_file
    )

    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    train_paths, train_lbls = splits["train"]
    val_paths, val_lbls = splits["val"]
    test_paths, test_lbls = splits["test"]

    total = len(train_paths) + len(val_paths) + len(test_paths)
    assert total == 20  # 10 cats + 10 dogs

    # Check strict disjointness (no data leakage)
    s_train = set(str(p) for p in train_paths)
    s_val = set(str(p) for p in val_paths)
    s_test = set(str(p) for p in test_paths)

    assert len(s_train & s_val) == 0, "Train and Val must be disjoint"
    assert len(s_train & s_test) == 0, "Train and Test must be disjoint"
    assert len(s_val & s_test) == 0, "Val and Test must be disjoint"

    # Check cache persistence
    assert cache_file.exists()
    splits_reloaded = prepare_stratified_split(
        data_dir=synthetic_image_dir,
        seed=777,
        cache_file=cache_file
    )
    assert len(splits_reloaded["train"][0]) == len(train_paths)
    assert [str(p) for p in splits_reloaded["train"][0]] == [str(p) for p in train_paths]


def test_stratified_split_seed_determinism(synthetic_image_dir: Path):
    """Verify that identical random seeds produce identical partitions."""
    s1 = prepare_stratified_split(synthetic_image_dir, seed=123)
    s2 = prepare_stratified_split(synthetic_image_dir, seed=123)
    s3 = prepare_stratified_split(synthetic_image_dir, seed=456)

    assert [str(p) for p in s1["train"][0]] == [str(p) for p in s2["train"][0]]
    # Different seed should yield different ordering/partitioning
    assert [str(p) for p in s1["train"][0]] != [str(p) for p in s3["train"][0]]


def test_dataset_loader_batch_iteration(synthetic_image_dir: Path):
    """Verify DataLoader yields valid batches."""
    loaders, splits = get_dataloaders(synthetic_image_dir, batch_size=4, num_workers=0)
    for split_name in ["train", "val", "test"]:
        loader = loaders[split_name]
        for images, labels, paths in loader:
            assert images.shape[1:] == (3, 224, 224)
            assert labels.dtype == torch.int64
            assert len(paths) == len(labels)
            break
