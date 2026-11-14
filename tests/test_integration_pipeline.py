"""
End-to-end integration tests:
1. Model training loop with synthetic data
2. Metric computation from model predictions
3. MetricRetrievalAdapter contrastive fine-tuning and inference
4. RetrievalFramework building and querying
5. Verification that existing artifacts and summaries load correctly into dashboard structures
"""

import json
from pathlib import Path

import numpy as np

from tcontext.evaluation.metrics import calculate_classification_metrics
from tcontext.models.cnn_models import MobileNetV3SmallClassifier
from tcontext.retrieval.retrieval_engine import (
    MetricRetrievalAdapter,
    RetrievalFramework,
)


def test_model_training_and_eval_integration(synthetic_dataloaders):
    """Test full training step and metric generation on lightweight model."""
    clf = MobileNetV3SmallClassifier(device="cpu", num_classes=2)
    history = clf.fit(
        train_loader=synthetic_dataloaders["train"],
        val_loader=synthetic_dataloaders["val"],
        epochs=1,
        verbose=False
    )
    assert "loss" in history
    assert "acc" in history

    test_loader = synthetic_dataloaders["test"]
    probs = clf.predict_proba(test_loader)
    preds = clf.predict(test_loader)
    labels = np.array([lbl for _, lbl, _ in test_loader.dataset])

    metrics = calculate_classification_metrics(labels, preds, probs)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_metric_retrieval_adapter_fit_and_transform():
    """Verify supervised contrastive metric learning adapter trains and transforms vectors."""
    adapter = MetricRetrievalAdapter(in_features=64, hidden_dim=64)

    rng = np.random.RandomState(42)
    embs = rng.randn(30, 64).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=-1, keepdims=True)
    labels = [0]*15 + [1]*15

    adapter.fit_adapter(embs, labels, epochs=2, batch_size=10, device="cpu")
    transformed = adapter.transform(embs, device="cpu")

    assert transformed.shape == (30, 64)
    # Vectors must remain L2 normalized
    norms = np.linalg.norm(transformed, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_retrieval_framework_build_and_query(temp_dir: Path):
    """Test full workflow of RetrievalFramework with ChromaDB and IMO."""
    rf = RetrievalFramework(persist_dir=temp_dir / "retrieval_db", top_k=3)

    dim = 64
    rng = np.random.RandomState(42)
    c0 = np.zeros(dim, dtype=np.float32); c0[:32] = 1.0; c0 /= np.linalg.norm(c0)
    c1 = np.zeros(dim, dtype=np.float32); c1[32:] = 1.0; c1 /= np.linalg.norm(c1)

    train_embs = np.vstack([
        c0 + 0.05 * rng.randn(10, dim).astype(np.float32),
        c1 + 0.05 * rng.randn(10, dim).astype(np.float32)
    ])
    train_embs /= np.linalg.norm(train_embs, axis=-1, keepdims=True)
    train_labels = [0]*10 + [1]*10
    train_paths = [f"path_{i}.jpg" for i in range(20)]

    build_res = rf.build_memory(train_embs, train_labels, train_paths, use_imo=True)
    assert build_res["indexed_count"] > 0

    test_embs = np.vstack([
        c0 + 0.05 * rng.randn(2, dim).astype(np.float32),
        c1 + 0.05 * rng.randn(2, dim).astype(np.float32)
    ])
    test_embs /= np.linalg.norm(test_embs, axis=-1, keepdims=True)
    preds, probs, neighbors, dists = rf.query(test_embs)

    assert len(preds) == 4
    assert probs.shape == (4, 2)
    assert len(neighbors) == 4
    assert len(dists) == 4


def test_artifacts_and_benchmark_summary_structure():
    """Verify production benchmark artifacts match required schemas."""
    summary_path = Path("tcontext/artifacts/benchmark_summary.json")
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert "metadata" in summary
        assert "dataset" in summary
        assert "primary_benchmark_table" in summary
        assert "cpu_benchmark_table" in summary
        assert "noise_experiment" in summary

        # Verify primary table has all 8 models
        table = summary["primary_benchmark_table"]
        assert len(table) == 8
        model_names = [r["model_name"] for r in table]
        assert "Proposed Framework (CLIP + ChromaDB + IMO)" in model_names
        assert "MobileNetV3-Small" in model_names
        assert "CLIP (ViT-B/32)" in model_names
