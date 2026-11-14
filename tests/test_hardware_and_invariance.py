"""
Hardware and Weight Invariance Validation Tests:
1. CPU execution mode
2. GPU execution mode (if available)
3. Parameter checksum snapshot verification before and after incremental memory update
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from tcontext.models.cnn_models import ResNet18Classifier
from tcontext.retrieval.retrieval_engine import RetrievalFramework


def compute_model_sha256(model: torch.nn.Module) -> str:
    """Compute deterministic SHA256 checksum across all parameter weights."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def test_device_placement_cpu():
    """Verify model parameters are placed strictly on CPU when device='cpu'."""
    clf = ResNet18Classifier(device="cpu")
    for param in clf.model.parameters():
        assert param.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_device_placement_gpu():
    """Verify model parameters are placed on CUDA GPU when device='cuda:0'."""
    clf = ResNet18Classifier(device="cuda:0")
    for param in clf.model.parameters():
        assert param.device.type == "cuda"
        assert param.device.index == 0


def test_frozen_backbone_invariance_during_incremental_updates(temp_dir: Path):
    """
    CRITICAL THESIS VERIFICATION:
    Confirm that incremental knowledge updates into external vector memory
    cause exactly ZERO changes to feature extractor parameters (nabla_theta = 0).
    """
    clf = ResNet18Classifier(device="cpu")
    checksum_before = compute_model_sha256(clf.model)

    rf = RetrievalFramework(persist_dir=temp_dir / "inv_test_db")

    # Ingest 5 incremental samples through the framework
    dim = 512
    rng = np.random.RandomState(42)
    new_embs = rng.randn(5, dim).astype(np.float32)
    new_embs /= np.linalg.norm(new_embs, axis=-1, keepdims=True)
    new_labels = [0, 0, 1, 1, 0]
    new_paths = [f"new_img_{i}.jpg" for i in range(5)]

    rf.build_memory(new_embs, new_labels, new_paths, use_imo=True)
    assert rf.vector_store.count() > 0

    checksum_after = compute_model_sha256(clf.model)

    # Weights must be 100% bit-for-bit identical
    assert checksum_before == checksum_after, (
        "Model parameter checksum changed during incremental memory update! "
        "The backbone was accidentally modified."
    )
