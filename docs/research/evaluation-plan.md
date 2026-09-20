# Evaluation Plan

Reports are generated from actual runs (`evaluation/reports/`); expected numbers are never hard-coded.
All benchmarks are on SYNTHETIC data.

## KYC (Phase 4; implemented)

Benchmark: `evaluation/benchmarks/kyc_benchmark.py`. Balanced category mix (clean 20%, typo, abbreviation,
transliteration, middle-name, name-order swap, address mismatch, DOB conflict), 6% duplicate entities,
6% namesakes (same name, different person: the hard negatives).

**Gold**: all customers sharing the query customer's `entity_id` (duplicates are gold; namesakes are not).

**Protocol**: split by *entity* into train 50% / dev 20% / test 30%, so no entity crosses splits. The
reranker is fit on train; each system's decision threshold is the F1-maximiser on dev; **all reported
numbers are test**. 95% CIs come from bootstrapping queries (1000 resamples).

**Metrics** (`evaluation/metrics/matching.py`), for a system returning up to K=20 scored candidates:

| Metric | Definition |
|---|---|
| precision / recall / F1 | pair-level, micro-averaged; TP = accepted gold, FP = accepted non-gold, FN = gold not accepted (including never retrieved) |
| missed-match rate | FN / (TP + FN) = 1 - recall |
| false-match rate | share of *queries* with at least one accepted non-gold candidate. A pair-level FPR is not comparable across systems because the number of evaluated negatives differs. |
| MRR, top-1 | rank of the first gold candidate (threshold-free) |
| candidate recall | share of gold retrieved into the top-K at all |
| latency | ms per query (shared signals + system scoring), single CPU process |

Also reported per variation type and for two slices: `duplicate_entity` and `namesake_negative`.

**Systems compared** (RQ1): exact, fuzzy, BM25, BM25-char, dense, hybrid, hybrid+reranker,
hybrid+structured, full (hybrid+reranker+structured). `hybrid+structured` isolates the effect of
structured verification from the reranker.

Run: `make kyc-benchmark` (about 5 minutes on the dev machine).

## Anomaly detection, LLM/RAG, agents, guardrails

Defined in Phases 5 onwards.
