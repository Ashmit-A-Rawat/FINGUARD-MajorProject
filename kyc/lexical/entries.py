"""Searchable name entries: each customer contributes its primary and alternate names."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from data_pipeline.consolidation.models import CanonicalCustomer


@dataclass
class NameEntries:
    texts: list[str]  # canonical name strings
    tokens: list[list[str]]
    starts: np.ndarray  # index of each customer's first entry (customers are contiguous)

    @classmethod
    def from_customers(cls, customers: Sequence[CanonicalCustomer]) -> "NameEntries":
        texts: list[str] = []
        starts: list[int] = []
        for customer in customers:
            starts.append(len(texts))
            names = [customer.name.canonical, *(a.canonical for a in customer.alternate_names)]
            texts.extend(dict.fromkeys(names))  # de-duplicate, keep order
        return cls(texts, [t.split() for t in texts], np.array(starts, dtype=np.int64))

    def customer_max(self, entry_scores: np.ndarray) -> np.ndarray:
        """Best entry score per customer."""
        result: np.ndarray = np.maximum.reduceat(entry_scores, self.starts)
        return result
