import numpy as np
import pytest

from evaluation.benchmarks.kyc_benchmark import KYCBenchmark
from kyc.entity_matcher import KYCEntityMatcher
from kyc.models import MatcherConfig, QueryIdentity
from kyc.reranking.reranker import LearnedReranker
from kyc.signals import SignalStore
from kyc.systems import SYSTEMS, CandidateScorer


@pytest.fixture(scope="module")
def scorer(benchmark: KYCBenchmark) -> CandidateScorer:
    return CandidateScorer(benchmark.customers, MatcherConfig(top_k=10))


@pytest.fixture(scope="module")
def store(benchmark: KYCBenchmark, embedder) -> SignalStore:  # type: ignore[no-untyped-def]
    return SignalStore(benchmark.customers, embedder)


def test_signals_are_bounded_and_one_per_customer(
    benchmark: KYCBenchmark, store: SignalStore
) -> None:
    q = benchmark.queries[0]
    (sig,) = store.compute([q.document.name.canonical])
    for arr in (sig.bm25, sig.bm25_char, sig.dense, sig.fuzzy):
        assert arr.shape == (len(benchmark.customers),)
        assert arr.min() >= 0.0 and arr.max() <= 1.0 + 1e-6


def test_clean_query_finds_its_own_customer_at_rank_one(
    benchmark: KYCBenchmark, store: SignalStore, scorer: CandidateScorer
) -> None:
    q = next(
        q for q in benchmark.queries if q.variation_type == "none" and not q.has_namesake_negative
    )
    (sig,) = store.compute([q.document.name.canonical])
    ids = [
        benchmark.customers[c.customer_idx].customer.customer_id
        for c in scorer.score("hybrid_structured", QueryIdentity.from_document(q.document), sig)
    ]
    assert ids[0] in q.gold_customer_ids


def test_reranked_systems_require_a_fitted_reranker(
    benchmark: KYCBenchmark, store: SignalStore, scorer: CandidateScorer
) -> None:
    q = benchmark.queries[0]
    (sig,) = store.compute([q.document.name.canonical])
    with pytest.raises(RuntimeError):
        scorer.score("full", QueryIdentity.from_document(q.document), sig)
    with pytest.raises(ValueError):
        scorer.score("nonsense", QueryIdentity.from_document(q.document), sig)


def test_reranker_needs_both_classes() -> None:
    with pytest.raises(ValueError):
        LearnedReranker().fit(np.zeros((4, 3)), np.zeros(4))
    with pytest.raises(RuntimeError):
        LearnedReranker().predict(np.zeros((1, 3)))


def test_every_system_runs_and_returns_sorted_candidates(
    benchmark: KYCBenchmark, store: SignalStore
) -> None:
    scorer = CandidateScorer(benchmark.customers, MatcherConfig(top_k=10))
    feats: list[np.ndarray] = []
    labels: list[bool] = []
    ids = [c.customer.customer_id for c in benchmark.customers]
    train = benchmark.split("train")[:80]
    for q, sig in zip(
        train, store.compute([q.document.name.canonical for q in train]), strict=True
    ):
        idx, matrix = scorer.training_candidates(QueryIdentity.from_document(q.document), sig)
        feats.append(matrix)
        labels.extend(ids[i] in q.gold_customer_ids for i in idx)
    scorer.reranker = LearnedReranker(0).fit(np.vstack(feats), np.array(labels, dtype=int))
    q = benchmark.split("test")[0]
    (sig,) = store.compute([q.document.name.canonical])
    for system in SYSTEMS:
        scored = scorer.score(system, QueryIdentity.from_document(q.document), sig)
        finals = [c.final_score for c in scored]
        assert finals == sorted(finals, reverse=True), system
        assert all(0.0 <= f <= 1.0 + 1e-9 for f in finals), system


def test_exact_baseline_only_matches_identical_name_and_dob(
    benchmark: KYCBenchmark, store: SignalStore, scorer: CandidateScorer
) -> None:
    varied = next(q for q in benchmark.queries if q.variation_type == "typo")
    (sig,) = store.compute([varied.document.name.canonical])
    assert scorer.score("exact", QueryIdentity.from_document(varied.document), sig) == []
    clean = next(q for q in benchmark.queries if q.variation_type == "none")
    (sig,) = store.compute([clean.document.name.canonical])
    hits = scorer.score("exact", QueryIdentity.from_document(clean.document), sig)
    assert hits and all(h.final_score == 1.0 for h in hits)


def test_matcher_output_exposes_all_required_evidence(benchmark: KYCBenchmark, embedder) -> None:  # type: ignore[no-untyped-def]
    matcher = KYCEntityMatcher(benchmark.customers, embedder)
    q = next(q for q in benchmark.queries if q.variation_type == "dob_conflict")
    results = matcher.match(q.document, top_n=3)
    assert 1 <= len(results) <= 3
    top = results[0]
    fields = set(top.model_dump())
    assert {
        "candidate_id",
        "lexical_score",
        "semantic_score",
        "reranker_score",
        "structured_match_score",
        "final_score",
        "confidence",
        "match_reasons",
        "contradictory_evidence",
    } <= fields
    assert top.structured_match_score is not None and top.reranker_score is None
    assert (
        any(c.startswith("dob_mismatch") for c in top.contradictory_evidence)
        or top.candidate_id not in q.gold_customer_ids
    )
    assert [r.final_score for r in results] == sorted(
        (r.final_score for r in results), reverse=True
    )


def test_confidence_bands_follow_threshold(benchmark: KYCBenchmark, embedder) -> None:  # type: ignore[no-untyped-def]
    m = KYCEntityMatcher(benchmark.customers, embedder, MatcherConfig(match_threshold=0.6))
    assert (
        m._confidence(0.75) == "high"
        and m._confidence(0.65) == "medium"
        and m._confidence(0.5) == "low"
    )
