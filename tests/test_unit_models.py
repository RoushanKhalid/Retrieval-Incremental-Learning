"""
Unit tests for neural network model wrappers: MobileNetV3-Small, EfficientNetB0, ResNet18, ViT-B/16, and CLIP.
Tests parameter counts, frozen backbones, model checkpoint sizes, and output tensor dimensions.
"""

from tcontext.models.cnn_models import (
    EfficientNetB0Classifier,
    MobileNetV3SmallClassifier,
    ResNet18Classifier,
)
from tcontext.models.vit_model import ViTClassifier


def test_mobilenet_initialization_and_frozen_backbone():
    """Verify MobileNetV3-Small has frozen backbone and trainable linear probe head."""
    clf = MobileNetV3SmallClassifier(device="cpu", num_classes=2)
    params = clf.get_param_count()
    assert params["total"] > 900_000
    assert params["trainable"] == 1154  # Head weights: (576 * 2) + 2 = 1154
    assert params["frozen"] == params["total"] - params["trainable"]

    # Model size MB should be > 0
    size_mb = clf.get_model_size_mb()
    assert 2.0 < size_mb < 6.0


def test_resnet18_initialization_and_frozen_backbone():
    """Verify ResNet18 has frozen residual backbone and trainable head."""
    clf = ResNet18Classifier(device="cpu", num_classes=2)
    params = clf.get_param_count()
    assert params["total"] > 11_000_000
    assert params["trainable"] == 1026  # Head weights: (512 * 2) + 2 = 1026
    assert params["frozen"] == params["total"] - params["trainable"]


def test_efficientnet_initialization_and_frozen_backbone():
    """Verify EfficientNetB0 parameter configuration."""
    clf = EfficientNetB0Classifier(device="cpu", num_classes=2)
    params = clf.get_param_count()
    assert params["total"] > 4_000_000
    assert params["trainable"] == 2562  # Head weights: (1280 * 2) + 2 = 2562


def test_vit_initialization():
    """Verify ViT-B/16 architecture."""
    clf = ViTClassifier(device="cpu", num_classes=2)
    params = clf.get_param_count()
    assert params["total"] > 85_000_000
    assert params["trainable"] == 1538  # (768 * 2) + 2 = 1538


def test_model_forward_pass_and_predictions(synthetic_dataloaders):
    """Verify inference pipeline produces valid probability distribution [N, 2]."""
    clf = MobileNetV3SmallClassifier(device="cpu", num_classes=2)
    val_loader = synthetic_dataloaders["val"]

    probs = clf.predict_proba(val_loader)
    preds = clf.predict(val_loader)

    assert probs.shape == (16, 2)
    assert preds.shape == (16,)
    # Probabilities must sum to 1.0 along class dimension
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)
    assert np.all((preds == 0) | (preds == 1))

import numpy as np
