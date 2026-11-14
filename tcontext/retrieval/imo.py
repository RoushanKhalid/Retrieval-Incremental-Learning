"""
Intelligent Memory Orchestrator (IMO).
The central thesis contribution: Controls vector memory quality via:
1. Prototype Consistency Gating
2. Label Conflict Detection
3. Quarantine Buffer
4. Outlier Filtering
5. Evidence Fusion (Base Parametric Model + Non-Parametric Retrieval)
"""

import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class IntelligentMemoryOrchestrator:
    """Intelligent Memory Orchestrator (IMO) for continuous knowledge curation."""

    def __init__(
        self,
        prototype_consistency_threshold: float = 0.65,
        confidence_gate_threshold: float = 0.75,
        outlier_distance_threshold: float = 0.45,
        fusion_lambda: float = 0.50,
        enable_prototype_check: bool = True,
        enable_conflict_check: bool = True,
        enable_outlier_filter: bool = True,
        enable_evidence_fusion: bool = True,
    ):
        self.tau_c = prototype_consistency_threshold
        self.conf_gate = confidence_gate_threshold
        self.outlier_thresh = outlier_distance_threshold
        self.fusion_lambda = fusion_lambda

        self.enable_prototype_check = enable_prototype_check
        self.enable_conflict_check = enable_conflict_check
        self.enable_outlier_filter = enable_outlier_filter
        self.enable_evidence_fusion = enable_evidence_fusion

        # Dynamic class prototype vectors: class_idx -> normalized 1D array
        self.prototypes: Dict[int, np.ndarray] = {}
        self.prototype_counts: Dict[int, int] = {}

        # Quarantine buffer for suspicious, conflicting, or outlier samples
        self.quarantine_buffer: List[Dict[str, Any]] = []

        # Orchestration audit statistics
        self.stats = {
            "presented": 0,
            "accepted": 0,
            "quarantined_conflict": 0,
            "quarantined_prototype": 0,
            "quarantined_outlier": 0,
            "rejected_confidence": 0,
        }

    def update_prototype(self, embedding: np.ndarray, label: int):
        """Incrementally update the normalized class prototype."""
        emb = embedding.reshape(-1)
        emb = emb / (np.linalg.norm(emb) + 1e-12)

        if label not in self.prototypes:
            self.prototypes[label] = emb.copy()
            self.prototype_counts[label] = 1
        else:
            n = self.prototype_counts[label]
            # Running weighted sum
            updated = self.prototypes[label] * n + emb
            self.prototype_counts[label] = n + 1
            self.prototypes[label] = updated / (np.linalg.norm(updated) + 1e-12)

    def evaluate_candidate(
        self,
        embedding: np.ndarray,
        candidate_label: int,
        vector_store_query_fn=None,
        classifier_confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Evaluate candidate sample through the IMO gating pipeline before memory insertion.
        Returns (is_accepted, reason).
        """
        self.stats["presented"] += 1
        emb = embedding.reshape(-1)
        emb = emb / (np.linalg.norm(emb) + 1e-12)

        # Gate 1: Optional Classifier Confidence Gate
        if classifier_confidence is not None and classifier_confidence < self.conf_gate:
            self.stats["rejected_confidence"] += 1
            reason = f"Classifier confidence {classifier_confidence:.3f} < {self.conf_gate:.2f}"
            self._quarantine(emb, candidate_label, reason, metadata)
            return False, reason

        # Gate 2: Class Prototype Consistency (Absolute and Relative Margin)
        if self.enable_prototype_check and candidate_label in self.prototypes:
            proto = self.prototypes[candidate_label]
            cos_sim = float(np.dot(emb, proto))
            if cos_sim < self.tau_c:
                self.stats["quarantined_prototype"] += 1
                reason = f"Prototype consistency {cos_sim:.3f} < threshold {self.tau_c:.2f}"
                self._quarantine(emb, candidate_label, reason, metadata)
                return False, reason

            # Relative class prototype margin check
            if len(self.prototypes) > 1:
                other_sims = [
                    float(np.dot(emb, p))
                    for lbl, p in self.prototypes.items()
                    if lbl != candidate_label
                ]
                if other_sims:
                    max_other = max(other_sims)
                    if cos_sim < max_other:
                        self.stats["quarantined_prototype"] += 1
                        reason = f"Prototype margin conflict: claimed similarity {cos_sim:.3f} < competing prototype {max_other:.3f}"
                        self._quarantine(emb, candidate_label, reason, metadata)
                        return False, reason

        # Gate 3 & 4: Nearest Neighbor Conflict and Outlier Check
        if (self.enable_conflict_check or self.enable_outlier_filter) and vector_store_query_fn:
            try:
                results = vector_store_query_fn(emb.reshape(1, -1), n_results=5)
                metas = results["metadatas"][0] if results and "metadatas" in results else []
                dists = results["distances"][0] if results and "distances" in results else []

                if metas:
                    nn_labels = [int(m["label"]) for m in metas]

                    # Conflict check: do nearest memory items disagree with candidate?
                    if self.enable_conflict_check and len(nn_labels) >= 3:
                        majority_label = Counter(nn_labels).most_common(1)[0][0]
                        if majority_label != candidate_label:
                            self.stats["quarantined_conflict"] += 1
                            reason = f"Memory conflict: majority neighbor label {majority_label} != candidate {candidate_label}"
                            self._quarantine(emb, candidate_label, reason, metadata)
                            return False, reason

                    # Outlier check: is average cosine distance too large?
                    if self.enable_outlier_filter and dists:
                        mean_dist = float(np.mean(dists))
                        if mean_dist > self.outlier_thresh:
                            self.stats["quarantined_outlier"] += 1
                            reason = f"Outlier detected: mean neighbor distance {mean_dist:.3f} > {self.outlier_thresh:.2f}"
                            self._quarantine(emb, candidate_label, reason, metadata)
                            return False, reason
            except Exception:
                pass

        # Candidate passed all gates: accept and update prototype
        self.stats["accepted"] += 1
        self.update_prototype(emb, candidate_label)
        return True, "Accepted"

    def _quarantine(self, embedding: np.ndarray, label: int, reason: str, metadata: Optional[Dict[str, Any]]):
        """Route failed candidate to quarantine buffer."""
        record = {
            "timestamp": time.time(),
            "label": label,
            "reason": reason,
            "metadata": metadata or {},
        }
        self.quarantine_buffer.append(record)

    def fuse_predictions(
        self,
        base_probs: np.ndarray,
        retrieval_neighbor_labels: List[List[int]],
        retrieval_distances: List[List[float]],
        temperature: float = 10.0,
        num_classes: int = 2
    ) -> np.ndarray:
        """
        Dynamically fuse base parametric classifier probabilities and retrieval evidence posteriors:
        p_final(c|x) = lambda * p_base(c|x) + (1 - lambda) * p_ret(c|x)
        """
        n_samples = len(base_probs)
        ret_probs = np.zeros((n_samples, num_classes), dtype=np.float32)

        for i in range(n_samples):
            labels = retrieval_neighbor_labels[i]
            dists = retrieval_distances[i]

            if not labels:
                ret_probs[i] = base_probs[i]
                continue

            # Convert cosine distances to similarities: sim = 1.0 - dist
            sims = np.array([1.0 - d for d in dists], dtype=np.float32)
            # Softmax weighting over retrieved neighbors
            weights = np.exp(sims * temperature)
            weights_sum = np.sum(weights) + 1e-12
            weights = weights / weights_sum

            for lbl, w in zip(labels, weights):
                if 0 <= lbl < num_classes:
                    ret_probs[i, lbl] += w

        if not self.enable_evidence_fusion:
            return ret_probs

        lam = self.fusion_lambda
        fused = lam * base_probs + (1.0 - lam) * ret_probs
        # Normalize across classes
        fused = fused / (np.sum(fused, axis=-1, keepdims=True) + 1e-12)
        return fused

    def get_summary(self) -> Dict[str, Any]:
        """Return IMO quality filtering metrics and status."""
        total = max(1, self.stats["presented"])
        return {
            "presented_samples": self.stats["presented"],
            "accepted_samples": self.stats["accepted"],
            "acceptance_rate": self.stats["accepted"] / total,
            "quarantined_total": len(self.quarantine_buffer),
            "reasons": {
                "prototype_mismatch": self.stats["quarantined_prototype"],
                "memory_conflict": self.stats["quarantined_conflict"],
                "outlier_density": self.stats["quarantined_outlier"],
                "confidence_gate": self.stats["rejected_confidence"],
            },
            "tracked_classes": list(self.prototypes.keys()),
        }
