from collections.abc import Sequence

import numpy as np
import pytest

from evaluation.benchmarks.kyc_benchmark import KYCBenchmark, build_kyc_benchmark


class HashingEmbedder:
    """Test-only deterministic embedder (hashed character trigrams). NOT a real dense model."""

    name = "test-hashing-embedder"

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            padded = f"^{text}$"
            for i in range(len(padded) - 2):
                out[row, hash(padded[i : i + 3]) % self.dim] += 1.0  # noqa: S324
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        normalised: np.ndarray = out / np.maximum(norms, 1e-9)
        return normalised


@pytest.fixture(scope="session", autouse=True)
def _fixed_hash_seed() -> None:
    # str hashes are randomised per process; the embedder only needs to be self-consistent
    # within one run, which it is. Nothing to do; this fixture documents that assumption.
    return None


@pytest.fixture(scope="session")
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture(scope="session")
def benchmark() -> KYCBenchmark:
    return build_kyc_benchmark(n_customers=300, seed=5)
