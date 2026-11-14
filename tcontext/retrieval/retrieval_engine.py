"""
Retrieval Engine comparing Naive Vector Retrieval against IMO-enhanced Retrieval.
"""

import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .imo import IntelligentMemoryOrchestrator
from .vector_store import ChromaVectorStore


class RetrievalFramework:
    """Manages vector memory build, query, and IMO-enhanced knowledge integration."""

    def __init__(
        self,
        persist_dir: Path | str,
        collection_name: str = "clip_image_embeddings",
        imo: Optional[IntelligentMemoryOrchestrator] = None,
        top_k: int = 5
    ):
        self.vector_store = ChromaVectorStore(persist_dir, collection_name=collection_name)
        self.imo = imo or IntelligentMemoryOrchestrator()
        self.top_k = top_k

    def build_memory(
        self,
        embeddings: np.ndarray,
        labels: List[int],
        paths: List[Path | str],
        use_imo: bool = True
    ) -> Dict[str, Any]:
        """
        Build or populate vector memory with candidate embeddings.
        If use_imo is True, runs each sample through the IMO gating pipeline.
        """
        t0 = time.time()
        accepted_ids = []
        accepted_embs = []
        accepted_metas = []

        # If not using IMO, accept everything directly
        if not use_imo:
            for i in range(len(labels)):
                doc_id = f"naive_train_{i:06d}"
                accepted_ids.append(doc_id)
                accepted_embs.append(embeddings[i])
                accepted_metas.append({
                    "label": int(labels[i]),
                    "path": str(paths[i]),
                    "curated_by": "naive"
                })
        else:
            # First pass: establish initial prototypes on clear bootstrap samples
            for i in range(min(50, len(labels))):
                self.imo.update_prototype(embeddings[i], labels[i])

            for i in range(len(labels)):
                emb = embeddings[i]
                lbl = int(labels[i])
                meta = {"path": str(paths[i])}

                is_accepted, reason = self.imo.evaluate_candidate(
                    embedding=emb,
                    candidate_label=lbl,
                    vector_store_query_fn=lambda q, n_results: self.vector_store.query(q, n_results=n_results) if self.vector_store.count() >= 5 else None,
                    metadata=meta
                )

                if is_accepted:
                    doc_id = f"imo_train_{i:06d}"
                    accepted_ids.append(doc_id)
                    accepted_embs.append(emb)
                    accepted_metas.append({
                        "label": lbl,
                        "path": str(paths[i]),
                        "curated_by": "imo"
                    })

        if accepted_ids:
            self.vector_store.add(accepted_ids, accepted_embs, accepted_metas)

        build_time = time.time() - t0
        return {
            "build_time_seconds": float(build_time),
            "indexed_count": len(accepted_ids),
            "total_candidates": len(labels),
            "imo_summary": self.imo.get_summary() if use_imo else None
        }

    def query(
        self,
        query_embeddings: np.ndarray,
        base_probs: Optional[np.ndarray] = None,
        use_evidence_fusion: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[List[int]], List[List[float]]]:
        """
        Query vector memory for nearest neighbors.
        Returns (predictions, probabilities, neighbor_labels, neighbor_distances).
        """
        results = self.vector_store.query(
            query_embeddings=query_embeddings,
            n_results=self.top_k,
            include=["metadatas", "distances"]
        )

        all_metas = results["metadatas"]
        all_dists = results["distances"]
        neighbor_labels = []
        neighbor_distances = []
        raw_retrieval_preds = []

        for row_metas, row_dists in zip(all_metas, all_dists):
            labels = [int(item["label"]) for item in row_metas]
            dists = [float(d) for d in row_dists]
            neighbor_labels.append(labels)
            neighbor_distances.append(dists)
            if labels:
                pred = Counter(labels).most_common(1)[0][0]
            else:
                pred = 0
            raw_retrieval_preds.append(pred)

        raw_retrieval_preds = np.array(raw_retrieval_preds, dtype=int)

        # Compute retrieval probabilities
        n_samples = len(query_embeddings)
        num_classes = 2
        ret_probs = np.zeros((n_samples, num_classes), dtype=np.float32)

        for i in range(n_samples):
            labels = neighbor_labels[i]
            dists = neighbor_distances[i]
            if labels:
                sims = np.array([max(0.0, 1.0 - d) for d in dists], dtype=np.float32)
                weights = np.exp(sims * 10.0)
                weights = weights / (np.sum(weights) + 1e-12)
                for lbl, w in zip(labels, weights):
                    if 0 <= lbl < num_classes:
                        ret_probs[i, lbl] += w
            else:
                ret_probs[i] = [0.5, 0.5]

        if use_evidence_fusion and base_probs is not None:
            fused_probs = self.imo.fuse_predictions(
                base_probs=base_probs,
                retrieval_neighbor_labels=neighbor_labels,
                retrieval_distances=neighbor_distances,
                num_classes=num_classes
            )
            final_preds = np.argmax(fused_probs, axis=-1)
            return final_preds, fused_probs, neighbor_labels, neighbor_distances

        return raw_retrieval_preds, ret_probs, neighbor_labels, neighbor_distances
