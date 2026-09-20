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

## Reconciliation (Phase 6; implemented)

Runner: `experiments/reconciliation/run_reconciliation_eval.py` (`make reconciliation-eval`). Ground truth is
the generator's `ledger_labels.csv` (six injected discrepancy types, one per transaction). Rules REC-001..008
map onto them; per-type and transaction-level precision/recall are reported, plus misattributed types and any
core-rule finding on an unlabelled transaction. **This checks that the rules and the injector agree; it is a
correctness check, not a measure of real-world quality** (both are deterministic and the tolerances were set with the
data ranges in view). Independent evidence is the hand-built edge cases in `backend/tests/reconciliation/`
(tolerance boundaries, duplicate ordering, float noise, currency+amount together, immutability).

## Knowledge-base retrieval (Phase 7; implemented)

Runner: `experiments/rag/run_retrieval_benchmark.py` (`make rag-benchmark`, about 1 minute).
Golden set: `evaluation/golden_dataset/kb_retrieval_questions.json` (50 hand-written questions: 16 direct, 12 paraphrased,
10 case-style scenarios, 5 rule-id lookups, 7 unanswerable). Adversarial set: `evaluation/adversarial_dataset/kb_injection*`.
Gold = (document, section); any listed gold section counts as a hit.

| Metric | Definition |
|---|---|
| hit@k | a gold section appears among the top-k chunks (section level); doc hit@k ignores the section |
| MRR, nDCG@5 | rank of the first gold hit; binary-relevance nDCG (each gold section counted once) |
| abstention AUROC | how well the top-1 score separates answerable from unanswerable questions |
| injection tripwire | recall of flagged adversarial documents and false-positive chunks among trusted ones |
| poison exposure | for topically matching untrusted documents: in top-5, outranks gold, top-1; with and without the trust filter |

95% CIs resample questions (n=43). Limits: one author wrote documents and questions, so vocabulary overlap may flatter lexical
retrieval; n is small; the 7 unanswerable questions make the abstention result fragile.

## Guardrails (Phase 10; implemented; supports RQ4)

Runner: `experiments/guardrails/run_guardrail_eval.py` (`make guardrail-eval`, needs the saved EXP-LLM-01 report).

- **Part A, claim-perturbation benchmark (ground truth by construction).** Claims are derived by template from real case evidence (so they are supported), then corrupted: number changed, invented id, swapped currency,
  swapped rule id, shifted date, wrong citation, no citation, unknown evidence id. Metrics: false-alarm rate on the supported claims and detection rate per perturbation type. It checks that the implementation
  does what it says; the author wrote both the templates and the perturbations, so it does **not** measure performance on free-form model text.
- **Part B, the guardrails on real model outputs.** Claim verdict counts (supported / unsupported / unverifiable), validated share, how often the advisory decision is raised, and problem cases still CLEAR after the guardrail.
  The unsupported flags were **audited by hand** (each read against its evidence) to estimate precision.
- RQ4 ("does self-critique reduce unsupported claims?") is only partly answerable here: the guardrails *detect and quarantine* unsupported claims (and raise the advisory decision); they do not make the model generate fewer of
  them. The with/without comparison at scale belongs to Phase 13.

## LLM grounding, fine-tuning, full ablations

Defined in Phases 12-13.
