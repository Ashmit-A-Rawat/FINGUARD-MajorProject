from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np


class HashingEmbedder:
    """Test-only deterministic embedder. NOT a real dense model."""

    name = "test-hashing-embedder"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import zlib

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                out[row, zlib.crc32(word.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        normalised: np.ndarray = out / np.maximum(norms, 1e-9)
        return normalised


class FakeClock:
    """Deterministic clock: each call advances one second."""

    def __init__(self) -> None:
        self._now = datetime(2025, 7, 1, 9, 0, 0)

    def __call__(self) -> datetime:
        self._now += timedelta(seconds=1)
        return self._now
