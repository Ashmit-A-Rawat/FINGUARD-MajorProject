"""Retrieval metrics over ranked chunk lists. A hit = a retrieved chunk whose (document, section)
is one of the question's gold sections (document-level variants ignore the section)."""

import math

import numpy as np
from sklearn.metrics import roc_auc_score

Gold = set[tuple[str, str]]


def first_hit_rank(
    ranked: list[tuple[str, str]], gold: Gold, *, doc_level: bool = False
) -> int | None:
    """1-based rank of the first relevant result, or None."""
    gold_docs = {d for d, _ in gold}
    for rank, (doc, section) in enumerate(ranked, start=1):
        if (doc in gold_docs) if doc_level else ((doc, section) in gold):
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank is not None else 0.0


def ndcg_at_k(ranked: list[tuple[str, str]], gold: Gold, k: int) -> float:
    """Binary-relevance nDCG; each gold section counts once (duplicate chunks of it do not)."""
    seen: set[tuple[str, str]] = set()
    gains = []
    for doc, section in ranked[:k]:
        relevant = (doc, section) in gold and (doc, section) not in seen
        gains.append(1.0 if relevant else 0.0)
        seen.add((doc, section))
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of the mean, resampling QUESTIONS."""
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def separation_auroc(answerable_scores: list[float], unanswerable_scores: list[float]) -> float:
    """AUROC for telling answerable from unanswerable questions by their top-1 score."""
    labels = [1] * len(answerable_scores) + [0] * len(unanswerable_scores)
    return float(roc_auc_score(labels, [*answerable_scores, *unanswerable_scores]))
