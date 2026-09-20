"""Exact-match baseline: identical canonical name AND identical date of birth."""

from collections import defaultdict
from collections.abc import Sequence
from datetime import date

from data_pipeline.consolidation.models import CanonicalCustomer


class ExactIndex:
    def __init__(self, customers: Sequence[CanonicalCustomer]) -> None:
        self._index: dict[tuple[str, date], set[int]] = defaultdict(set)
        for i, customer in enumerate(customers):
            dob = customer.customer.date_of_birth
            for name in [customer.name, *customer.alternate_names]:
                self._index[(name.canonical, dob)].add(i)

    def lookup(self, canonical_name: str, dob: date | None) -> list[int]:
        if dob is None:
            return []
        return sorted(self._index.get((canonical_name, dob), ()))
