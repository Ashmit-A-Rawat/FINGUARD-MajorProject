"""Fuzzy string similarity over name entries (rapidfuzz token-sort ratio)."""

import numpy as np
from rapidfuzz import fuzz, process


def fuzzy_scores(query: str, entry_texts: list[str]) -> np.ndarray:
    """Similarity in [0, 1] between the query and every entry."""
    matrix = process.cdist([query], entry_texts, scorer=fuzz.token_sort_ratio, dtype=np.float32)
    row: np.ndarray = matrix[0] / 100.0
    return row
