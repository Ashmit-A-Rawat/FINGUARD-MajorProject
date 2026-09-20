"""Pairwise name features for the learned reranker. Name-only: no DOB or address (stage G's job)."""

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from data_pipeline.consolidation.models import CanonicalCustomer
from data_pipeline.normalization.models import NormalizedName

FEATURE_NAMES = (
    "bm25",
    "bm25_char",
    "dense",
    "fuzzy",
    "hybrid",
    "inverse_rank",
    "jw_full",
    "jw_first",
    "jw_last",
    "token_set",
    "first_compatible",
    "first_exact",
    "last_exact",
    "middle_compatible",
    "query_has_initials",
    "query_reordered",
    "token_count_gap",
)


def _initial_compatible(a: str, b: str) -> bool:
    return a == b or ((len(a) == 1 or len(b) == 1) and a[:1] == b[:1])


def _middle_compatible(query: NormalizedName, candidate: NormalizedName) -> float:
    if not query.middle or not candidate.middle:
        return 1.0  # a missing middle name is not evidence against a match
    return float(_initial_compatible(query.middle[0], candidate.middle[0]))


def _name_pair_features(query: NormalizedName, candidate: NormalizedName) -> list[float]:
    return [
        JaroWinkler.normalized_similarity(query.canonical, candidate.canonical),
        JaroWinkler.normalized_similarity(query.first, candidate.first),
        JaroWinkler.normalized_similarity(query.last, candidate.last),
        fuzz.token_set_ratio(query.canonical, candidate.canonical) / 100.0,
        float(_initial_compatible(query.first, candidate.first)),
        float(query.first == candidate.first),
        float(query.last == candidate.last),
        _middle_compatible(query, candidate),
        float(query.has_initials),
        float(query.reordered),
        float(abs(len(query.tokens) - len(candidate.tokens))),
    ]


def build_feature_row(
    query: NormalizedName,
    candidate: CanonicalCustomer,
    signals: tuple[float, float, float, float, float],
    rank: int,
) -> np.ndarray:
    """``signals`` = (bm25, bm25_char, dense, fuzzy, hybrid); ``rank`` is 1-based hybrid rank.

    Name-structure features use the candidate name (primary or alternate) closest to the query.
    """
    options = [candidate.name, *candidate.alternate_names]
    best = max(
        options, key=lambda n: JaroWinkler.normalized_similarity(query.canonical, n.canonical)
    )
    return np.array([*signals, 1.0 / rank, *_name_pair_features(query, best)], dtype=np.float64)
