"""
Unit tests for ChromaDB Vector Store wrapper and HNSW index operations.
"""

from pathlib import Path

import numpy as np

from tcontext.retrieval.vector_store import ChromaVectorStore


def test_vector_store_crud(temp_dir: Path):
    """Test Create, Add, Query, Count, and Reset in isolated temporary ChromaDB."""
    db_path = temp_dir / "test_chroma_db"
    store = ChromaVectorStore(db_path, collection_name="unit_test_col")

    assert store.count() == 0

    dim = 512
    v1 = np.zeros(dim); v1[0] = 1.0
    v2 = np.zeros(dim); v2[1] = 1.0

    store.add(
        ids=["id_1", "id_2"],
        embeddings=[v1, v2],
        metadatas=[{"label": 0, "tag": "first"}, {"label": 1, "tag": "second"}]
    )

    assert store.count() == 2

    # Query with vector identical to v1
    res = store.query(query_embeddings=[v1], n_results=1)
    assert res["ids"][0][0] == "id_1"
    assert res["metadatas"][0][0]["label"] == 0
    assert res["distances"][0][0] < 1e-4

    # Test directory footprint
    size_mb = store.get_storage_size_mb()
    assert size_mb > 0.0

    # Test reset
    store.reset()
    assert store.count() == 0


def test_vector_store_persistence(temp_dir: Path):
    """Verify vectors persist across store client re-initialization."""
    db_path = temp_dir / "persistent_db"
    store1 = ChromaVectorStore(db_path, collection_name="persist_col")

    dim = 512
    v = np.random.randn(dim)
    v = (v / np.linalg.norm(v)).tolist()

    store1.add(ids=["p_1"], embeddings=[v], metadatas=[{"label": 1}])
    assert store1.count() == 1
    del store1

    # Reload client pointing to same folder
    store2 = ChromaVectorStore(db_path, collection_name="persist_col")
    assert store2.count() == 1
    res = store2.query(query_embeddings=[v], n_results=1)
    assert res["ids"][0][0] == "p_1"
