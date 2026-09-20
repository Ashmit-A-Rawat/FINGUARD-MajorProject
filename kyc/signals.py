"""First-stage retrieval signals for a query, computed once and shared by all systems."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from data_pipeline.consolidation.models import CanonicalCustomer
from kyc.lexical.bm25 import BM25Index, char_trigrams
from kyc.lexical.entries import NameEntries
from kyc.lexical.fuzzy import fuzzy_scores
from kyc.semantic.dense import DenseIndex
from kyc.semantic.embedder import Embedder


@dataclass
class QuerySignals:
    """Each array has one score in [0, 1] per customer (best over that customer's names)."""

    bm25: np.ndarray
    bm25_char: np.ndarray
    dense: np.ndarray
    fuzzy: np.ndarray


class SignalStore:
    def __init__(self, customers: Sequence[CanonicalCustomer], embedder: Embedder) -> None:
        self.embedder = embedder
        self.entries = NameEntries.from_customers(customers)
        self._bm25 = BM25Index(self.entries.tokens)
        self._bm25_char = BM25Index([char_trigrams(t) for t in self.entries.texts])
        self._dense = DenseIndex(embedder.encode(self.entries.texts))

    def compute(self, names: Sequence[str]) -> list[QuerySignals]:
        """``names`` are canonical (normalised) query names; embeddings are batched."""
        vectors = self.embedder.encode(names)
        out: list[QuerySignals] = []
        for name, vector in zip(names, vectors, strict=True):
            e = self.entries
            out.append(
                QuerySignals(
                    bm25=e.customer_max(self._bm25.normalized_scores(name.split())),
                    bm25_char=e.customer_max(
                        self._bm25_char.normalized_scores(char_trigrams(name))
                    ),
                    dense=e.customer_max(self._dense.scores(vector)),
                    fuzzy=e.customer_max(fuzzy_scores(name, e.texts)),
                )
            )
        return out
