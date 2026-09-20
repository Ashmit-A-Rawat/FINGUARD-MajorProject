"""BM25 over name tokens, with scores normalised to [0, 1] so thresholds are comparable.

Implemented directly (about 50 lines) rather than through a library so the normalisation
by the query's own ceiling score is explicit and testable.
"""

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np


def word_tokens(text: str) -> list[str]:
    return text.split()


def char_trigrams(text: str) -> list[str]:
    """Character trigrams per word with boundary markers; robust to typos."""
    grams: list[str] = []
    for word in text.split():
        padded = f"^{word}$"
        grams.extend(padded[i : i + 3] for i in range(max(1, len(padded) - 2)))
    return grams


class BM25Index:
    def __init__(
        self, documents: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.k1, self.b = k1, b
        self.n = len(documents)
        self.lengths = np.array([len(d) for d in documents], dtype=np.float64)
        self.avgdl = float(self.lengths.mean()) or 1.0
        raw: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(documents):
            for term, tf in Counter(doc).items():
                raw[term].append((i, tf))
        self.idf = {
            term: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for term, p in raw.items()
        }
        self.postings = {
            term: (np.array([i for i, _ in p]), np.array([tf for _, tf in p], dtype=np.float64))
            for term, p in raw.items()
        }
        self._unseen_idf = math.log(1 + (self.n + 0.5) / 0.5)

    def raw_scores(self, query: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.n)
        for term in set(query):
            if term not in self.postings:
                continue
            idx, tf = self.postings[term]
            denom = tf + self.k1 * (1 - self.b + self.b * self.lengths[idx] / self.avgdl)
            scores[idx] += self.idf[term] * tf * (self.k1 + 1) / denom
        return scores

    def self_score(self, query: Sequence[str]) -> float:
        """Score of a document identical to the query: the ceiling used for normalisation.

        Unseen query terms still count towards the ceiling, so a query full of unknown
        tokens can never look like a perfect match.
        """
        counts = Counter(query)
        norm = self.k1 * (1 - self.b + self.b * len(query) / self.avgdl)
        return sum(
            self.idf.get(term, self._unseen_idf) * tf * (self.k1 + 1) / (tf + norm)
            for term, tf in counts.items()
        )

    def normalized_scores(self, query: Sequence[str]) -> np.ndarray:
        ceiling = self.self_score(query)
        if ceiling <= 0:
            return np.zeros(self.n)
        result: np.ndarray = np.clip(self.raw_scores(query) / ceiling, 0.0, 1.0)
        return result
