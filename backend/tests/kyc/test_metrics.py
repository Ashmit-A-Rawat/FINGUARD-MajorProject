import numpy as np
import pytest

from evaluation.metrics.matching import (
    FlatOutcomes,
    RankedOutcome,
    best_threshold,
    bootstrap_ci,
    ranking_metrics,
    summarize,
)


def _outcomes() -> list[RankedOutcome]:
    return [
        RankedOutcome(frozenset({"a", "b"}), [("a", 0.9), ("x", 0.7), ("b", 0.4)], "c1"),
        RankedOutcome(frozenset({"c"}), [("y", 0.8), ("c", 0.6)], "c2"),
        RankedOutcome(frozenset({"d"}), [("z", 0.3)], "c2"),  # gold never retrieved
    ]


def test_counts_and_summary_at_threshold() -> None:
    flat = FlatOutcomes(_outcomes())
    tp, fp, fn = flat.counts(0.5)
    # q1: accept a,x -> tp1 fp1 fn1 ; q2: accept y,c -> tp1 fp1 fn0 ; q3: none -> fn1
    assert tp.tolist() == [1, 1, 0] and fp.tolist() == [1, 1, 0] and fn.tolist() == [1, 0, 1]
    m = summarize(tp, fp, fn)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(2 / 4)
    assert m["f1"] == pytest.approx(0.5)
    assert m["missed_match_rate"] == pytest.approx(0.5)
    assert m["false_match_rate"] == pytest.approx(2 / 3)  # queries with >=1 false match


def test_threshold_is_inclusive() -> None:
    flat = FlatOutcomes(_outcomes())
    assert flat.counts(0.9)[0].tolist() == [1, 0, 0]


def test_best_threshold_maximises_f1_on_perfectly_separable_data() -> None:
    perfect = [
        RankedOutcome(frozenset({"a"}), [("a", 0.9), ("x", 0.2)], "c"),
        RankedOutcome(frozenset({"b"}), [("b", 0.8), ("y", 0.1)], "c"),
    ]
    t = best_threshold(FlatOutcomes(perfect))
    assert summarize(*FlatOutcomes(perfect).counts(t))["f1"] == 1.0
    assert 0.2 < t <= 0.8


def test_ranking_metrics() -> None:
    m = ranking_metrics(_outcomes())
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert m["top1_accuracy"] == pytest.approx(1 / 3)
    assert m["candidate_recall_any"] == pytest.approx(2 / 3)
    assert m["candidate_recall_pairs"] == pytest.approx(3 / 4)


def test_empty_acceptance_gives_zero_not_error() -> None:
    m = summarize(*FlatOutcomes(_outcomes()).counts(2.0))
    assert m["precision"] == 0 and m["f1"] == 0 and m["false_match_rate"] == 0


def test_bootstrap_ci_brackets_point_estimate_and_is_reproducible() -> None:
    tp = np.array([1.0] * 80 + [0.0] * 20)
    fp = np.array([0.0] * 90 + [1.0] * 10)
    fn = 1 - tp
    ci = bootstrap_ci(tp, fp, fn, n_boot=300, seed=1)
    point = summarize(tp, fp, fn)["f1"]
    assert ci["f1"][0] <= point <= ci["f1"][1]
    assert ci == bootstrap_ci(tp, fp, fn, n_boot=300, seed=1)
