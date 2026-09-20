"""ChromaDB-backed vector store (cosine similarity)."""

import contextlib
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from knowledge_base.models import Chunk


def _metadata(chunk: Chunk) -> dict[str, Any]:
    return {
        "document_id": chunk.document_id,
        "title": chunk.title,
        "section": chunk.section,
        "section_id": chunk.section_id,
        "version": chunk.version,
        "source": chunk.source,
        "page": chunk.page if chunk.page is not None else -1,
        "chunk_index": chunk.chunk_index,
        "trust": chunk.trust,
        "injection_flags": "|".join(chunk.injection_flags),
    }


class ChromaVectorStore:
    """``path=None`` gives an in-memory store (tests); otherwise a persistent directory."""

    def __init__(self, path: Path | None, collection: str = "fin_guard_kb") -> None:
        self._client: Any = (
            chromadb.PersistentClient(str(path)) if path else chromadb.EphemeralClient()
        )
        self._name = collection
        self._collection: Any = None

    def reset(self) -> None:
        # chroma raises different errors when the collection does not exist yet
        with contextlib.suppress(Exception):
            self._client.delete_collection(self._name)
        self._collection = self._client.create_collection(
            self._name, metadata={"hnsw:space": "cosine"}
        )

    def open(self) -> None:
        self._collection = self._client.get_collection(self._name)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[_metadata(c) for c in chunks],
        )

    def count(self) -> int:
        return int(self._collection.count())

    def query(self, embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Top-k (chunk_id, cosine similarity)."""
        res = self._collection.query(
            query_embeddings=[embedding.tolist()], n_results=min(k, self.count())
        )
        return [
            (cid, 1.0 - float(d)) for cid, d in zip(res["ids"][0], res["distances"][0], strict=True)
        ]

    def embeddings_of(self, chunk_ids: list[str]) -> np.ndarray:
        got = self._collection.get(ids=chunk_ids, include=["embeddings"])
        by_id = dict(zip(got["ids"], got["embeddings"], strict=True))
        return np.array([by_id[i] for i in chunk_ids], dtype=np.float32)
