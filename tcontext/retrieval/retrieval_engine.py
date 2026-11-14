"""
Retrieval Engine comparing Naive Vector Retrieval against IMO-enhanced Retrieval.
"""

import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .imo import IntelligentMemoryOrchestrator
from .vector_store import ChromaVectorStore


class MetricRetrievalAdapter(nn.Module):
    """
    Contrastive Metric Learning Adapter for Fine-Tuning Retrieval Embeddings.
    Adapts the 512-dim visual representation so intra-class cosine similarities
    are maximized and inter-class distances are penalized.
    """

    def __init__(self, in_features: int = 512, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_features)
        )
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.adapter(x)
        out = x + self.scale * residual
        return F.normalize(out, dim=-1)

    def fit_adapter(
        self,
        train_embeddings: np.ndarray,
        train_labels: List[int],
        epochs: int = 15,
        lr: float = 0.001,
        batch_size: int = 64,
        device: str = "cpu"
    ) -> "MetricRetrievalAdapter":
        """Fine-tune the metric adapter on training embeddings using Supervised Contrastive Loss."""
        self.to(device)
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.0001)

        t_embs = torch.tensor(train_embeddings, dtype=torch.float32)
        t_lbls = torch.tensor(train_labels, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(t_embs, t_lbls)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch_embs, batch_lbls in loader:
                batch_embs, batch_lbls = batch_embs.to(device), batch_lbls.to(device)
                optimizer.zero_grad()
                proj = self(batch_embs)

                sim_mat = torch.matmul(proj, proj.T)
                labels_eq = (batch_lbls.unsqueeze(1) == batch_lbls.unsqueeze(0)).float()

                pos_sim = (sim_mat * labels_eq).sum() / (labels_eq.sum() + 1e-6)
                neg_mask = 1.0 - labels_eq
                neg_sim = (sim_mat * neg_mask).sum() / (neg_mask.sum() + 1e-6)

                loss = -pos_sim + 0.5 * neg_sim
                loss.backward()
                optimizer.step()

        self.eval()
        return self

    def transform(self, embeddings: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Project embeddings through fine-tuned metric space."""
        self.to(device)
        self.eval()
        with torch.no_grad():
            inp = torch.tensor(embeddings, dtype=torch.float32).to(device)
            out = self(inp).cpu().numpy()
        return out


def tune_retrieval_hyperparameters(
    val_embeddings: np.ndarray,
    val_labels: List[int],
    vector_store: ChromaVectorStore,
    k_candidates: List[int] = [1, 3, 5, 7, 9],
    base_probs: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Fine-tunes retrieval hyperparameters (k, weighting, alpha) on the validation partition.
    """
    best_k = 5
    best_acc = 0.0
    results = {}
    val_labels_arr = np.array(val_labels, dtype=int)

    for k in k_candidates:
        q_res = vector_store.query(val_embeddings, n_results=k)
        preds = []
        for metas in q_res["metadatas"]:
            votes = [int(m["label"]) for m in metas]
            p = Counter(votes).most_common(1)[0][0] if votes else 0
            preds.append(p)
        acc = float(np.mean(np.array(preds) == val_labels_arr))
        results[f"k_{k}"] = acc
        if acc > best_acc:
            best_acc = acc
            best_k = k

    best_alpha = 0.5
    best_fused_acc = best_acc
    if base_probs is not None:
        q_res = vector_store.query(val_embeddings, n_results=best_k)
        ret_probs = np.zeros((len(val_embeddings), 2), dtype=np.float32)
        for i, (metas, dists) in enumerate(zip(q_res["metadatas"], q_res["distances"])):
            for m, d in zip(metas, dists):
                sim = max(0.0, 1.0 - d)
                ret_probs[i, int(m["label"])] += sim
        row_sums = ret_probs.sum(axis=1, keepdims=True)
        ret_probs = np.where(row_sums > 0, ret_probs / row_sums, 0.5)

        for alpha in np.linspace(0.0, 1.0, 21):
            fused = alpha * base_probs + (1.0 - alpha) * ret_probs
            f_preds = np.argmax(fused, axis=-1)
            f_acc = float(np.mean(f_preds == val_labels_arr))
            if f_acc > best_fused_acc:
                best_fused_acc = f_acc
                best_alpha = float(alpha)

    return {
        "optimal_k": best_k,
        "best_k_val_accuracy": best_acc,
        "optimal_fusion_alpha": best_alpha,
        "best_fused_val_accuracy": best_fused_acc,
        "k_sweep": results
    }


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
