"""Semantic injection tripwire: a small logistic-regression classifier on sentence embeddings.

Why: the regular-expression tripwire (sanitize.py) catches known phrasings but generalises poorly
to new ones (EXP-ADV-02). This layer scores the MEANING of short free text (a payment memo, say).
It is trained by ``scripts/train_tripwire.py`` and stored as a tiny JSON file; it is only used with
the same embedding model it was trained with, and it is a TRIPWIRE like the regex layer: it flags
for a human and forces REVIEW, it does not delete text and does not replace the engine floor.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from knowledge_base.embeddings.embedder import Embedder

MODEL_PATH = Path(__file__).with_name("tripwire_model.json")
MIN_WORDS = 3  # single tokens (ids, currencies, dates) are never classified


class SemanticTripwire:
    def __init__(self, embedder: Embedder, spec: dict[str, Any]) -> None:
        self._embedder = embedder
        self._w = np.asarray(spec["weights"], dtype=np.float32)
        self._b = float(spec["bias"])
        self.threshold = float(spec["threshold"])
        self.spec = spec
        self._cache = lru_cache(maxsize=4096)(self._score_one)

    @classmethod
    def load(cls, embedder: Embedder, path: Path = MODEL_PATH) -> "SemanticTripwire | None":
        """None if there is no trained model or the embedder differs from the training one."""
        if not path.exists():
            return None
        spec = json.loads(path.read_text())
        return cls(embedder, spec) if embedder.name == spec["embedding_model"] else None

    def _score_one(self, text: str) -> float:
        vec = self._embedder.encode([text])[0]
        return float(1.0 / (1.0 + np.exp(-(float(vec @ self._w) + self._b))))

    def score(self, text: str) -> float:
        return self._cache(text)

    def is_injection(self, text: str) -> bool:
        return len(text.split()) >= MIN_WORDS and self.score(text) >= self.threshold
