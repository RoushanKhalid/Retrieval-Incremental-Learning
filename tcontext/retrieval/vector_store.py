"""
Vector Store Wrapper for ChromaDB.
Manages persistent and in-memory collections of normalized feature embeddings.
"""

from pathlib import Path
from typing import Any

import chromadb
import numpy as np


class ChromaVectorStore:
    """Interface to persistent or in-memory ChromaDB vector storage."""

    def __init__(
        self,
        persist_dir: Path | str,
        collection_name: str = "clip_image_embeddings",
        distance_metric: str = "cosine"
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric}
        )

    def reset(self):
        """Wipe collection and re-initialize."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric}
        )

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray | list[list[float]],
        metadatas: list[dict[str, Any]] | None = None
    ):
        """Add vectors with metadata to collection."""
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def query(
        self,
        query_embeddings: np.ndarray | list[list[float]],
        n_results: int = 5,
        include: list[str] | None = None
    ) -> dict[str, Any]:
        """Perform nearest neighbor cosine search."""
        if include is None:
            include = ["metadatas", "distances"]
        if isinstance(query_embeddings, np.ndarray):
            query_embeddings = query_embeddings.tolist()
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=include
        )

    def count(self) -> int:
        """Return total indexed vectors."""
        return self.collection.count()

    def get_storage_size_mb(self) -> float:
        """Calculate total directory footprint of the vector database."""
        total_bytes = 0
        for f in self.persist_dir.rglob("*"):
            if f.is_file():
                total_bytes += f.stat().st_size
        return total_bytes / (1024 ** 2)
