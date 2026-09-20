import numpy as np
import pytest

from anomaly_detection.temporal.splits import temporal_split
from evaluation.metrics.anomaly import (
    budget_metrics,
    cluster_bootstrap,
    f1_optimal_threshold,
    operating_point,
    pr_auc,
    recall_by_type,
    roc_auc,
)


def _times(n: int) -> np.ndarray:
    return np.datetime64("2025-01-01T00:00:00", "ns") + (np.arange(n) * 3600).astype(
        "timedelta64[s]"
    )


def test_split_is_chronological_and_exhaustive() -> None:
    ts, ep = _times(1000), np.array([""] * 1000)
    s = temporal_split(ts, ep)
    s.check_no_future_leakage(ts)
    assert len(s.train) + len(s.val) + len(s.test) == 1000 and s.purged_rows == 0
    assert abs(len(s.train) - 600) <= 2 and abs(len(s.test) - 200) <= 2


def test_straddling_episode_is_purged_from_all_splits() -> None:
    ts = _times(1000)
    ep = np.array([""] * 1000, dtype=object)
    ep[595:606] = "EP-X"  # crosses the train/val boundary near row 600
    ep[100:105] = "EP-SAFE"
    s = temporal_split(ts, ep.astype(str))
    used = set(s.train) | set(s.val) | set(s.test)
    assert not used & set(range(595, 606))
    assert set(range(100, 105)) <= set(s.train)
    assert s.purged_episodes == 1 and s.purged_rows == 11


def test_bad_fractions_rejected() -> None:
    with pytest.raises(ValueError):
        temporal_split(_times(10), np.array([""] * 10), (0.5, 0.2, 0.2))


def test_operating_point_hand_computed() -> None:
    y = np.array([1, 1, 0, 0, 0, 0])
    score = np.array([0.9, 0.4, 0.8, 0.3, 0.2, 0.1])
    m = operating_point(y, score, 0.5)
    assert (m["precision"], m["recall"]) == (0.5, 0.5)
    assert m["false_positive_rate"] == pytest.approx(1 / 4)
    assert m["false_negative_rate"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_auc_metrics_on_perfect_and_reversed_scores() -> None:
    y = np.array([0, 0, 1, 1])
    assert pr_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_f1_threshold_and_budget() -> None:
    y = np.array([1, 1, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.5, 0.3, 0.2, 0.1])
    t = f1_optimal_threshold(y, score)
    assert operating_point(y, score, t)["f1"] == 1.0
    b = budget_metrics(y, score, 2 / 6)
    assert b == {"precision": 1.0, "recall": 1.0}


def test_recall_by_type_excludes_normal_rows() -> None:
    types = np.array(["none", "burst", "burst", "geo_change"])
    out = recall_by_type(types, np.array([0.9, 0.9, 0.1, 0.9]), 0.5)
    assert set(out) == {"burst", "geo_change"} and out["burst"]["recall"] == 0.5


def test_cluster_bootstrap_is_reproducible_and_brackets_estimate() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    score = y * 0.5 + rng.random(2000) * 0.6
    clusters = rng.integers(0, 300, 2000)
    a = cluster_bootstrap(y, score, clusters, 0.6, n_boot=100, seed=3)
    assert a == cluster_bootstrap(y, score, clusters, 0.6, n_boot=100, seed=3)
    assert a["pr_auc"][0] <= pr_auc(y, score) <= a["pr_auc"][1]
