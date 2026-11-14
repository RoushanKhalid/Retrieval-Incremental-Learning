"""
Unit tests for the Intelligent Memory Orchestrator (IMO):
1. Prototype updates & running centroid normalization
2. Classifier Confidence Gate (Gate 1)
3. Prototype Consistency Gate (Gate 2)
4. Relative Prototype Margin Conflict Gate
5. Outlier & Nearest Neighbor Conflict Filtering
6. Evidence Fusion
"""

import numpy as np

from tcontext.retrieval.imo import IntelligentMemoryOrchestrator


def test_prototype_update_and_normalization():
    """Verify running prototype vector updates maintain unit norm."""
    imo = IntelligentMemoryOrchestrator()
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])

    imo.update_prototype(v1, label=0)
    assert np.allclose(imo.prototypes[0], [1.0, 0.0, 0.0])
    assert np.isclose(np.linalg.norm(imo.prototypes[0]), 1.0)

    imo.update_prototype(v2, label=0)
    expected = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    assert np.allclose(imo.prototypes[0], expected, atol=1e-5)
    assert np.isclose(np.linalg.norm(imo.prototypes[0]), 1.0)


def test_confidence_gate():
    """Verify Gate 1 rejects candidates with classifier confidence below threshold."""
    imo = IntelligentMemoryOrchestrator(confidence_gate_threshold=0.75)
    emb = np.array([1.0, 0.0, 0.0])

    accepted, reason = imo.evaluate_candidate(emb, candidate_label=0, classifier_confidence=0.70)
    assert not accepted
    assert "confidence" in reason.lower()
    assert imo.stats["rejected_confidence"] == 1
    assert len(imo.quarantine_buffer) == 1


def test_prototype_consistency_and_margin_conflict():
    """Verify Gate 2 rejects samples conflicting with class prototype or competing margins."""
    imo = IntelligentMemoryOrchestrator(prototype_consistency_threshold=0.65)
    dim = 64
    proto_cat = np.zeros(dim); proto_cat[:32] = 1.0; proto_cat /= np.linalg.norm(proto_cat)
    proto_dog = np.zeros(dim); proto_dog[32:] = 1.0; proto_dog /= np.linalg.norm(proto_dog)

    imo.update_prototype(proto_cat, 0)
    imo.update_prototype(proto_dog, 1)

    # Cat candidate with cat label -> should be accepted
    v_cat = proto_cat.copy()
    acc1, reason1 = imo.evaluate_candidate(v_cat, candidate_label=0)
    assert acc1 is True
    assert reason1 == "Accepted"

    # Cat candidate with dog label (label conflict) -> margin check should reject
    acc2, reason2 = imo.evaluate_candidate(v_cat, candidate_label=1)
    assert acc2 is False
    assert "prototype" in reason2.lower() or "margin" in reason2.lower()

    # Orthogonal noise vector -> consistency check should reject
    noise = np.random.randn(dim); noise /= np.linalg.norm(noise)
    acc3, reason3 = imo.evaluate_candidate(noise, candidate_label=0)
    assert acc3 is False


def test_evidence_fusion():
    """Verify Bayesian evidence fusion between parametric base model and retrieval posteriors."""
    imo = IntelligentMemoryOrchestrator(fusion_lambda=0.5, enable_evidence_fusion=True)

    base_probs = np.array([[0.8, 0.2]])  # Model leans cat (0)
    neighbor_labels = [[1, 1, 1, 1, 1]]  # Retrieval strongly dog (1)
    neighbor_dists = [[0.05, 0.05, 0.05, 0.05, 0.05]]

    fused = imo.fuse_predictions(
        base_probs=base_probs,
        retrieval_neighbor_labels=neighbor_labels,
        retrieval_distances=neighbor_dists,
        num_classes=2
    )

    # Retrieval should pull the distribution towards dog
    assert fused[0, 1] > 0.4
    assert np.isclose(fused.sum(axis=-1)[0], 1.0)
