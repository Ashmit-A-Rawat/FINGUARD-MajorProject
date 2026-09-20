"""Metrics for imbalanced anomaly detection.

Accuracy is deliberately absent. Threshold-free: PR-AUC (average precision) and ROC-AUC.
At an operating threshold (chosen on VALIDATION): precision, recall, F1, false-positive rate,
false-negative rate. Alert-budget metrics (precision/recall in the top x% of scores) need no
threshold at all and are label-free to apply, which makes unsupervised models comparable.

Confidence intervals resample CUSTOMERS (clusters), not rows, because a customer's transactions,
especially those in an anomaly episode, are correlated.
"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def pr_auc(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> float:
    return float(average_precision_score(y, score, sample_weight=weight))


def roc_auc(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> float:
    return float(roc_auc_score(y, score, sample_weight=weight))


def f1_optimal_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Score threshold that maximises F1 on the given (validation) data."""
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def operating_point(
    y: np.ndarray, score: np.ndarray, threshold: float, weight: np.ndarray | None = None
) -> dict[str, float]:
    w = np.ones(len(y)) if weight is None else weight
    flagged = score >= threshold
    tp = float((w * (flagged & (y == 1))).sum())
    fp = float((w * (flagged & (y == 0))).sum())
    fn = float((w * (~flagged & (y == 1))).sum())
    tn = float((w * (~flagged & (y == 0))).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "flagged_share": float((w * flagged).sum() / w.sum()),
    }


def budget_metrics(y: np.ndarray, score: np.ndarray, fraction: float) -> dict[str, float]:
    """Flag the top ``fraction`` of rows by score (deterministic tie-breaking)."""
    k = max(1, round(fraction * len(y)))
    top = np.argsort(-score, kind="stable")[:k]
    hits = float(y[top].sum())
    return {"precision": hits / k, "recall": hits / max(float(y.sum()), 1.0)}


def recall_by_type(
    types: np.ndarray, score: np.ndarray, threshold: float
) -> dict[str, dict[str, float]]:
    """Share of each anomaly type flagged at the threshold (normal rows excluded)."""
    out: dict[str, dict[str, float]] = {}
    for kind in sorted(set(types.tolist()) - {"none"}):
        mask = types == kind
        out[kind] = {"n": float(mask.sum()), "recall": float((score[mask] >= threshold).mean())}
    return out


def cluster_bootstrap(
    y: np.ndarray,
    score: np.ndarray,
    clusters: np.ndarray,
    threshold: float,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """95% percentile CIs for PR-AUC, ROC-AUC and F1, resampling clusters with replacement.

    A resample is expressed as integer row weights (times each cluster was drawn), which is
    exactly equivalent to concatenating the drawn clusters.
    """
    rng = np.random.default_rng(seed)
    _, cluster_index = np.unique(clusters, return_inverse=True)
    n_clusters = int(cluster_index.max()) + 1
    samples: dict[str, list[float]] = {"pr_auc": [], "roc_auc": [], "f1": []}
    for _ in range(n_boot):
        draws = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)
        weight = draws[cluster_index].astype(float)
        if (weight * y).sum() == 0:
            continue
        samples["pr_auc"].append(pr_auc(y, score, weight))
        samples["roc_auc"].append(roc_auc(y, score, weight))
        samples["f1"].append(operating_point(y, score, threshold, weight)["f1"])
    return {
        k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in samples.items()
    }
