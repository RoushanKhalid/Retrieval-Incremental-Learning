"""
Noise Robustness Benchmark.
Evaluates hypothesis H1: IMO-enabled retrieval significantly outperforms naive cache insertion under label noise.
Compares accuracy retention and quarantine behavior across 0%, 10%, 20%, 30% synthetic label corruption.
"""

import random
from pathlib import Path
from typing import Any

import numpy as np

from tcontext.evaluation.metrics import calculate_classification_metrics
from tcontext.retrieval.imo import IntelligentMemoryOrchestrator
from tcontext.retrieval.vector_store import ChromaVectorStore


def inject_label_noise(labels: list[int], noise_rate: float, seed: int = 777) -> list[int]:
    """Invert binary labels at a fixed noise rate."""
    if noise_rate <= 0.0:
        return list(labels)
    rng = random.Random(seed)
    noisy = list(labels)
    n_corrupt = int(len(labels) * noise_rate)
    indices_to_corrupt = rng.sample(range(len(labels)), n_corrupt)
    for idx in indices_to_corrupt:
        noisy[idx] = 1 - noisy[idx]
    return noisy


def run_noise_experiment(
    train_embeddings: np.ndarray,
    train_labels: list[int],
    train_paths: list[Path | str],
    test_embeddings: np.ndarray,
    test_labels: list[int],
    noise_rates: list[float] = [0.0, 0.10, 0.20, 0.30],
    seed: int = 777,
    temp_dir: Path | str = "tcontext/vector_db/noise_exp"
) -> dict[str, Any]:
    """
    Run noise corruption benchmark comparing Naive Retrieval vs IMO-enhanced Retrieval.
    """
    temp_path = Path(temp_dir)
    results: dict[str, list] = {
        "noise_rates": noise_rates,
        "naive_accuracy": [],
        "naive_f1": [],
        "imo_accuracy": [],
        "imo_f1": [],
        "imo_quarantined_count": [],
        "imo_acceptance_rate": [],
    }

    test_labels_arr = np.array(test_labels, dtype=int)

    for rate in noise_rates:
        noisy_train_labels = inject_label_noise(train_labels, noise_rate=rate, seed=seed)

        # 1. Benchmark Naive Retrieval
        naive_db_path = temp_path / f"naive_noise_{int(rate*100)}_{seed}"
        if naive_db_path.exists():
            import shutil
            try:
                shutil.rmtree(naive_db_path)
            except Exception:
                naive_db_path = temp_path / f"naive_noise_{int(rate*100)}_{seed}_{random.randint(1000,9999)}"
        naive_store = ChromaVectorStore(naive_db_path, collection_name="naive_noise")
        naive_ids = [f"naive_{i}" for i in range(len(noisy_train_labels))]
        naive_metas = [{"label": int(l)} for l in noisy_train_labels]
        naive_store.add(naive_ids, train_embeddings, naive_metas)

        query_res = naive_store.query(test_embeddings, n_results=5)
        naive_preds = []
        for metas in query_res["metadatas"]:
            votes = [int(m["label"]) for m in metas]
            from collections import Counter
            naive_preds.append(Counter(votes).most_common(1)[0][0] if votes else 0)
        naive_metrics = calculate_classification_metrics(test_labels_arr, np.array(naive_preds))
        results["naive_accuracy"].append(naive_metrics["accuracy"])
        results["naive_f1"].append(naive_metrics["f1"])

        # 2. Benchmark IMO-Enhanced Retrieval
        imo_db_path = temp_path / f"imo_noise_{int(rate*100)}_{seed}"
        if imo_db_path.exists():
            import shutil
            try:
                shutil.rmtree(imo_db_path)
            except Exception:
                imo_db_path = temp_path / f"imo_noise_{int(rate*100)}_{seed}_{random.randint(1000,9999)}"
        imo_store = ChromaVectorStore(imo_db_path, collection_name="imo_noise")
        imo = IntelligentMemoryOrchestrator(
            prototype_consistency_threshold=0.65,
            confidence_gate_threshold=0.75,
            outlier_distance_threshold=0.45
        )

        # Seed running class prototypes with clean bootstrap anchors
        bootstrap_n = min(40, len(train_labels))
        for k in range(bootstrap_n):
            imo.update_prototype(train_embeddings[k], train_labels[k])

        accepted_embs: list = []
        accepted_labels: list = []
        accepted_ids: list = []
        accepted_metas: list = []

        def in_memory_query(query_vec, n_results=5):
            if len(accepted_embs) < n_results:
                return None
            mat = np.array(accepted_embs)
            q = query_vec.reshape(-1)
            sims = np.dot(mat, q)
            top_indices = np.argsort(sims)[::-1][:n_results]
            return {
                "metadatas": [[{"label": accepted_labels[idx]} for idx in top_indices]],
                "distances": [[float(1.0 - sims[idx]) for idx in top_indices]]
            }

        for i in range(len(noisy_train_labels)):
            emb = train_embeddings[i]
            lbl = noisy_train_labels[i]
            meta = {"path": str(train_paths[i])}

            is_accepted, _ = imo.evaluate_candidate(
                embedding=emb,
                candidate_label=lbl,
                vector_store_query_fn=in_memory_query,
                metadata=meta
            )
            if is_accepted:
                accepted_embs.append(emb)
                accepted_labels.append(lbl)
                accepted_ids.append(f"imo_{i}")
                accepted_metas.append({"label": int(lbl)})

        if accepted_ids:
            imo_store.add(accepted_ids, accepted_embs, accepted_metas)

        imo_query_res = imo_store.query(test_embeddings, n_results=5)
        imo_preds = []
        for metas in imo_query_res["metadatas"]:
            votes = [int(m["label"]) for m in metas]
            from collections import Counter
            imo_preds.append(Counter(votes).most_common(1)[0][0] if votes else 0)
        imo_metrics = calculate_classification_metrics(test_labels_arr, np.array(imo_preds))

        results["imo_accuracy"].append(imo_metrics["accuracy"])
        results["imo_f1"].append(imo_metrics["f1"])
        results["imo_quarantined_count"].append(len(imo.quarantine_buffer))
        results["imo_acceptance_rate"].append(imo.stats["accepted"] / max(1, imo.stats["presented"]))

    return results
