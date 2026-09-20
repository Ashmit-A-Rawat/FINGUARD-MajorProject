from collections.abc import Sequence
from pathlib import Path

import numpy as np

DOCS = Path("knowledge_base/documents")
ADV = Path("evaluation/adversarial_dataset/kb_injection")


class HashingEmbedder:
    """Test-only deterministic embedder (hashed word + trigram counts). NOT a real dense model."""

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


def write_doc(path: Path, body: str, **front: str) -> Path:
    meta = {
        "document_id": "D-1",
        "title": "T",
        "version": "1",
        "source": "s",
        "effective_date": "2025-01-01",
    }
    meta.update(front)
    header = "\n".join(f"{k}: {v}" for k, v in meta.items())
    path.write_text(f"---\n{header}\n---\n{body}", encoding="utf-8")
    return path
