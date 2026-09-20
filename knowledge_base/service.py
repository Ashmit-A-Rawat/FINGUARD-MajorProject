"""Build, persist and open a knowledge-base index."""

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from knowledge_base.chunking.chunker import ChunkingConfig, chunk_document
from knowledge_base.embeddings.embedder import Embedder
from knowledge_base.ingestion.loader import LoadResult, Rejected, load_documents
from knowledge_base.models import Chunk, Trust
from knowledge_base.retrieval.retriever import Mode, Retriever
from knowledge_base.vector_store.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)
MANIFEST = "manifest.json"
CHUNKS = "chunks.jsonl"


class IndexMismatchError(RuntimeError):
    """The stored index was built with a different embedding model or is otherwise unusable."""


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    trust: Trust = "trusted"  # a property of where the files come from, not of their content


@dataclass
class KnowledgeBase:
    retriever: Retriever
    chunks: list[Chunk]
    rejected: list[Rejected]
    manifest: dict[str, object]

    @classmethod
    def build(
        cls,
        sources: list[SourceSpec],
        embedder: Embedder,
        chunking: ChunkingConfig | None = None,
        index_dir: Path | None = None,
        mode: Mode = "hybrid",
    ) -> "KnowledgeBase":
        """Ingest -> clean -> chunk -> embed -> store. ``index_dir=None`` stays in memory."""
        chunking = chunking or ChunkingConfig()
        loaded = LoadResult()
        for spec in sources:
            part = load_documents(spec.path, spec.trust)
            loaded.documents += part.documents
            loaded.rejected += part.rejected
        ids = [d.document_id for d in loaded.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("document ids collide across sources")
        chunks = [c for d in loaded.documents for c in chunk_document(d, chunking)]
        if not chunks:
            raise ValueError("no chunks produced; nothing to index")
        # In-memory clients are process-wide singletons, so each in-memory index needs its own
        # collection or building a second index would destroy the first.
        collection = "fin_guard_kb" if index_dir else f"kb_{uuid.uuid4().hex}"
        store = ChromaVectorStore(index_dir / "chroma" if index_dir else None, collection)
        store.reset()
        embeddings = embedder.encode([c.embed_text for c in chunks])
        store.add(chunks, embeddings)
        manifest: dict[str, object] = {
            "embedding_model": embedder.name,
            "chunking": json.loads(chunking.model_dump_json()),
            "n_documents": len(loaded.documents),
            "n_chunks": len(chunks),
            "documents": {
                d.document_id: {"version": d.version, "sha256": d.content_hash, "trust": d.trust}
                for d in loaded.documents
            },
            "rejected": [r.__dict__ for r in loaded.rejected],
            "flagged_chunks": [c.chunk_id for c in chunks if c.injection_flags],
            "chunks_sha256": hashlib.sha256(
                "".join(c.chunk_id + c.text for c in chunks).encode()
            ).hexdigest(),
        }
        if index_dir:
            index_dir.mkdir(parents=True, exist_ok=True)
            (index_dir / CHUNKS).write_text("\n".join(c.model_dump_json() for c in chunks))
            (index_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
        logger.info(
            "indexed %d chunks from %d documents (%d rejected files)",
            len(chunks),
            len(loaded.documents),
            len(loaded.rejected),
        )
        return cls(Retriever(chunks, store, embedder, mode), chunks, loaded.rejected, manifest)

    @classmethod
    def open(cls, index_dir: Path, embedder: Embedder, mode: Mode = "hybrid") -> "KnowledgeBase":
        manifest = json.loads((index_dir / MANIFEST).read_text())
        if manifest["embedding_model"] != embedder.name:
            raise IndexMismatchError(
                f"index built with {manifest['embedding_model']!r}, opened with {embedder.name!r}"
            )
        chunks = [
            Chunk.model_validate_json(line)
            for line in (index_dir / CHUNKS).read_text().splitlines()
        ]
        store = ChromaVectorStore(index_dir / "chroma")
        store.open()
        if store.count() != len(chunks):
            raise IndexMismatchError("vector store and chunk file disagree; rebuild the index")
        return cls(Retriever(chunks, store, embedder, mode), chunks, [], manifest)
