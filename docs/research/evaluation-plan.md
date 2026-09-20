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

## Anomaly detection (Phase 5; implemented)

Runner: `experiments/anomaly/run_anomaly_benchmark.py` (`make anomaly-benchmark`, about 3 minutes on the medium
preset). Protocol and leakage controls: [ml-architecture](../architecture/ml-architecture.md#transaction-anomaly-detection-anomaly_detection).

| Metric | Definition |
|---|---|
| PR-AUC | average precision (main metric; a random scorer scores about the base rate, 1.75% on test) |
| ROC-AUC | reported but not relied on under roughly 1:56 imbalance |
| precision / recall / F1 | at the F1-maximising threshold chosen on VALIDATION |
| FPR / FNR | share of normal rows flagged / share of anomalies missed at that threshold |
| P@k%, R@k% | precision and recall when flagging the top 1% / 2% of test rows by score (threshold-free) |
| recall by anomaly type | which injected anomaly types each model catches |
| latency | median ms per 1000 rows (batch) and per single row, single CPU process |

Accuracy is deliberately not reported. 95% CIs are a **customer-clustered bootstrap** (200 resamples): rows of one
customer, especially within an anomaly episode, are correlated, so resampling rows would understate uncertainty.

## LLM/RAG, agents, guardrails

Defined in Phases 7 onwards.
