import numpy as np


class DenseIndex:
    """Cosine similarity between a query embedding and every entry embedding."""

    def __init__(self, entry_embeddings: np.ndarray) -> None:
        self.embeddings = entry_embeddings

    def scores(self, query_embedding: np.ndarray) -> np.ndarray:
        similarity: np.ndarray = self.embeddings @ query_embedding
        clipped: np.ndarray = np.clip(similarity, 0.0, 1.0)
        return clipped
