"""Retriever: BM25, dense (Chroma) or hybrid (reciprocal rank fusion) over knowledge-base chunks."""

from collections.abc import Sequence
from typing import Literal

import numpy as np

from knowledge_base.embeddings.embedder import Embedder
from knowledge_base.models import Chunk, RetrievedChunk
from knowledge_base.retrieval.tokenizer import tokenize
from knowledge_base.vector_store.chroma_store import ChromaVectorStore
from kyc.lexical.bm25 import BM25Index

Mode = Literal["bm25", "dense", "hybrid"]
RRF_K = 60  # standard reciprocal-rank-fusion constant


class Retriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        store: ChromaVectorStore,
        embedder: Embedder,
        mode: Mode = "hybrid",
        candidate_k: int = 30,
    ) -> None:
        self.chunks = list(chunks)
        self._by_id = {c.chunk_id: c for c in self.chunks}
        self.store, self.embedder, self.mode, self.candidate_k = store, embedder, mode, candidate_k
        self._bm25 = BM25Index([tokenize(c.embed_text) for c in self.chunks])

    def _bm25_scores(self, query: str) -> np.ndarray:
        return self._bm25.normalized_scores(tokenize(query))

    def _dense_ranked(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        return self.store.query(query_vec, k)

    def search(
        self, query: str, k: int = 5, mode: Mode | None = None, include_untrusted: bool = False
    ) -> list[RetrievedChunk]:
        """Top-k chunks. Chunks from UNTRUSTED sources are excluded unless asked for explicitly:
        ranking cannot protect against a topically matching poisoned document (see the
        poisoning experiment), so trust is enforced here, by source, before ranking."""
        mode = mode or self.mode
        index_of = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        allowed = np.array([include_untrusted or c.trust == "trusted" for c in self.chunks])
        bm25 = np.where(allowed, self._bm25_scores(query), -1.0)
        query_vec = self.embedder.encode([query])[0]
        fetch = min(len(self.chunks), max(k, self.candidate_k) + int((~allowed).sum()))
        dense_top = {
            cid: score
            for cid, score in self._dense_ranked(query_vec, fetch)
            if allowed[index_of[cid]]
        }

        def bm25_order(limit: int) -> list[int]:
            order = np.argsort(-bm25, kind="stable")
            return [int(i) for i in order if allowed[i]][:limit]

        if mode == "bm25":
            ranked = [(self.chunks[i].chunk_id, float(bm25[i])) for i in bm25_order(k)]
        elif mode == "dense":
            ranked = sorted(dense_top.items(), key=lambda kv: -kv[1])[:k]
        else:  # hybrid: reciprocal rank fusion of the two candidate lists
            fused: dict[str, float] = {}
            for rank, i in enumerate(bm25_order(max(k, self.candidate_k)), start=1):
                cid = self.chunks[i].chunk_id
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            dense_order = sorted(dense_top.items(), key=lambda kv: -kv[1])
            for rank, (cid, _) in enumerate(dense_order, start=1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

        ids = [cid for cid, _ in ranked]
        missing = [cid for cid in ids if cid not in dense_top]
        extra: dict[str, float] = {}
        if missing:  # candidates only one retriever proposed still get an exact cosine score
            vectors = self.store.embeddings_of(missing)
            extra = dict(zip(missing, (vectors @ query_vec).tolist(), strict=True))
        results: list[RetrievedChunk] = []
        for rank, (cid, score) in enumerate(ranked, start=1):
            chunk = self._by_id[cid]
            results.append(
                RetrievedChunk(
                    chunk_id=cid,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=score,
                    rank=rank,
                    metadata={
                        "document_id": chunk.document_id,
                        "source": chunk.source,
                        "section": chunk.section,
                        "section_id": chunk.section_id,
                        "page": chunk.page,
                        "version": chunk.version,
                        "title": chunk.title,
                        "trust": chunk.trust,
                        "injection_flags": chunk.injection_flags,
                    },
                    component_scores={
                        "bm25": float(max(bm25[index_of[cid]], 0.0)),
                        "dense": float(dense_top.get(cid, extra.get(cid, 0.0))),
                    },
                )
            )
        return results
