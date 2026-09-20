"""Decision-level and ranking metrics for entity matching.

Definitions (also in docs/research/evaluation-plan.md):
* A system returns up to K scored candidates per query; a candidate is ACCEPTED if score >= t.
* Pair-level TP/FP/FN: TP = accepted gold; FP = accepted non-gold; FN = gold not accepted
  (including gold never retrieved). precision, recall, F1 are micro-averaged over pairs.
* missed_match_rate = FN / (TP + FN)  (= 1 - recall).
* false_match_rate  = share of QUERIES with at least one accepted non-gold candidate.
  (A pair-level FPR is not comparable across systems because the number of evaluated
  negatives differs by system.)
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankedOutcome:
    gold: frozenset[str]
    candidates: list[tuple[str, float]]  # (customer_id, score), best first
    category: str
    slices: tuple[str, ...] = ()


class FlatOutcomes:
    """Vectorised view of outcomes so a threshold sweep is cheap."""

    def __init__(self, outcomes: list[RankedOutcome]) -> None:
        self.n = len(outcomes)
        scores: list[float] = []
        is_gold: list[bool] = []
        owner: list[int] = []
        for q, outcome in enumerate(outcomes):
            for cid, score in outcome.candidates:
                scores.append(score)
                is_gold.append(cid in outcome.gold)
                owner.append(q)
        self.scores = np.array(scores)
        self.is_gold = np.array(is_gold, dtype=bool)
        self.owner = np.array(owner, dtype=np.int64)
        self.gold_count = np.array([len(o.gold) for o in outcomes], dtype=np.int64)

    def counts(self, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-query (tp, fp, fn) at ``threshold``."""
        accepted = self.scores >= threshold
        tp = np.bincount(self.owner, weights=accepted & self.is_gold, minlength=self.n)
        fp = np.bincount(self.owner, weights=accepted & ~self.is_gold, minlength=self.n)
        return tp, fp, self.gold_count - tp


def summarize(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> dict[str, float]:
    total_tp, total_fp, total_fn = float(tp.sum()), float(fp.sum()), float(fn.sum())
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missed_match_rate": 1.0 - recall if total_tp + total_fn else 0.0,
        "false_match_rate": float((fp > 0).mean()) if len(fp) else 0.0,
    }


def best_threshold(flat: FlatOutcomes, grid: np.ndarray | None = None) -> float:
    """Threshold maximising F1 (largest such threshold on ties, i.e. the stricter one)."""
    grid = np.linspace(0.0, 1.0, 201) if grid is None else grid
    best_t, best_f1 = float(grid[-1]), -1.0
    for t in grid[::-1]:
        f1 = summarize(*flat.counts(float(t)))["f1"]
        if f1 > best_f1 + 1e-12:
            best_t, best_f1 = float(t), f1
    return best_t


def ranking_metrics(outcomes: list[RankedOutcome]) -> dict[str, float]:
    """Threshold-free retrieval quality over the returned candidate lists."""
    reciprocal, top1, any_gold, gold_found, gold_total = [], [], [], 0, 0
    for o in outcomes:
        ranks = [i for i, (cid, _) in enumerate(o.candidates, start=1) if cid in o.gold]
        reciprocal.append(1.0 / ranks[0] if ranks else 0.0)
        top1.append(1.0 if ranks and ranks[0] == 1 else 0.0)
        any_gold.append(1.0 if ranks else 0.0)
        gold_found += len(ranks)
        gold_total += len(o.gold)
    n = max(len(outcomes), 1)
    return {
        "mrr": sum(reciprocal) / n,
        "top1_accuracy": sum(top1) / n,
        "candidate_recall_any": sum(any_gold) / n,  # at least one gold candidate retrieved
        "candidate_recall_pairs": gold_found / gold_total if gold_total else 0.0,
    }


def bootstrap_ci(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """95% percentile CIs, resampling QUERIES with replacement."""
    rng = np.random.default_rng(seed)
    n = len(tp)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples: dict[str, list[float]] = {}
    for row in idx:
        for key, value in summarize(tp[row], fp[row], fn[row]).items():
            samples.setdefault(key, []).append(value)
    return {
        key: (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        for key, vals in samples.items()
    }
