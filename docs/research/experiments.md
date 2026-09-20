# Experiments

Each entry names the script, seed and the generated artifact. Numbers below are copied from the
generated reports in `evaluation/reports/`; regenerate rather than edit. All data is SYNTHETIC.

## EXP-KYC-01: KYC retrieval / verification ablation (RQ1)

- Script: `experiments/kyc/run_kyc_benchmark.py --n-customers 4000 --seed {42,43}` (`make kyc-benchmark`)
- Artifacts: `evaluation/reports/kyc/kyc_benchmark_n4000_seed42.{json,md}` and `..._seed43.{json,md}`
- Protocol: [evaluation plan](evaluation-plan.md). Seed 42 was the primary run; seed 43 is an
  independent replication run afterwards, with no code changes between them.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`, CPU.

| system | F1 seed 42 [95% CI] | F1 seed 43 [95% CI] | MRR 42 / 43 | candidate recall (pairs) 42 / 43 |
|---|---|---|---|---|
| exact | 0.631 [0.605, 0.658] | 0.640 [0.615, 0.662] | 0.500 / 0.512 | 0.461 / 0.470 |
| fuzzy | 0.645 [0.622, 0.665] | 0.659 [0.640, 0.680] | 0.819 / 0.824 | 0.988 / 0.983 |
| BM25 (word) | 0.558 [0.537, 0.580] | 0.577 [0.556, 0.599] | 0.709 / 0.724 | 0.920 / 0.918 |
| BM25 (char trigram) | 0.558 [0.537, 0.580] | 0.574 [0.554, 0.594] | 0.801 / 0.798 | 0.988 / 0.986 |
| dense | 0.558 [0.538, 0.579] | 0.581 [0.559, 0.602] | 0.740 / 0.755 | 0.949 / 0.957 |
| hybrid (BM25 + dense) | 0.560 [0.539, 0.582] | 0.580 [0.559, 0.602] | 0.747 / 0.754 | 0.953 / 0.961 |
| hybrid + reranker | 0.714 [0.693, 0.735] | 0.735 [0.715, 0.757] | 0.835 / 0.853 | 0.953 / 0.961 |
| hybrid + structured | 0.971 [0.964, 0.977] | 0.977 [0.970, 0.983] | 0.971 / 0.978 | 0.953 / 0.961 |
| full (hybrid + reranker + structured) | 0.963 [0.955, 0.971] | 0.959 [0.951, 0.967] | 0.964 / 0.969 | 0.953 / 0.961 |

### What the data supports (both seeds agree)

1. **Hybrid BM25 + dense did not beat its parts on this benchmark.** Hybrid F1 (0.560 / 0.580)
   is indistinguishable from BM25 or dense alone, and below fuzzy (0.645 / 0.659). Its MRR is
   below char-trigram BM25 and fuzzy, and its candidate recall (0.953 / 0.961) is below both of
   those (about 0.99). So **RQ1 is not supported for this hybrid design**. Caveats: alpha=0.5 was fixed
   a priori and not tuned, and word-BM25 and MiniLM are both weak on typos.
2. **The learned name reranker helped clearly** on F1 (0.560 to 0.714, and 0.580 to 0.735; CIs do not
   overlap), but at the cost of a higher per-query false-match rate (0.30) and about 19 ms extra latency per query.
3. **Structured verification (DOB + address) made the largest difference**: F1 0.71 to 0.97. Precision
   is about 1.0 because namesakes differ in DOB/address.
4. **The reranker added nothing on top of structured verification**: `full` is at or slightly below
   `hybrid+structured` (CIs overlap or `full` lower), with more latency. Do not claim the reranker helps
   in the full pipeline.
5. **Fixed first stage limits recall**: top-K by fused score misses about 4-5% of gold pairs. Char-trigram
   BM25 and fuzzy retrieve about 99%. A candidate *union* of several retrievers would likely fix this.
   That is a design change made after seeing test results, so it must be evaluated on a fresh seed
   before being claimed as an improvement.

### Threats to validity (read before quoting numbers)

- **Structured verification is unrealistically decisive here.** DOB is uniformly random and the
  address pool is small, so DOB agreement almost uniquely identifies a person. Precision of about 1.0
  reflects the synthetic data, not real-world KYC.
- **The name pool is tiny (about 50 first x 50 last names)**, so many different people have
  near-identical names. That inflates name-only false matches and depresses name-only F1. The
  *ranking* of methods is more trustworthy than the absolute values.
- Name-only systems' dev-chosen thresholds are at about 0.99 (they accept essentially only identical names),
  so several lexical systems have near-identical F1; MRR and candidate recall separate them better.
- Variations come from fixed generator rules; real transliteration and typo patterns differ.
- Single embedding model, single reranker type, single alpha. No hyperparameter search was run.
- The reranker is fit and evaluated within the same synthetic generator, which favours it.

## EXP-ANOM-01: Supervised vs unsupervised vs neural anomaly detection (RQ2)

- Script: `experiments/anomaly/run_anomaly_benchmark.py --preset medium --seed {42,43}` (`make anomaly-benchmark`)
- Data: synthetic medium preset, 250,000 transactions, 10,000 customers, 2% injected anomalies. Seed 43 uses a
  freshly generated, independent dataset (also a different training seed); no code changes between runs.
- Artifacts: `evaluation/reports/anomaly/anomaly_benchmark_medium_seed{42,43}.{json,md}`
- Protocol: [ml-architecture](../architecture/ml-architecture.md#transaction-anomaly-detection-anomaly_detection) and
  [evaluation plan](evaluation-plan.md). Chronological 60/20/20, test base rate 1.75-1.77%, thresholds from validation.

| model | PR-AUC seed 42 [95% CI] | PR-AUC seed 43 [95% CI] | F1 42 / 43 | ROC-AUC 42 / 43 |
|---|---|---|---|---|
| logistic_regression | 0.537 [0.459, 0.594] | 0.572 [0.506, 0.637] | 0.477 / 0.499 | 0.972 / 0.974 |
| random_forest | 0.804 [0.765, 0.831] | 0.814 [0.780, 0.844] | 0.710 / 0.715 | 0.988 / 0.989 |
| xgboost | 0.794 [0.756, 0.826] | 0.825 [0.790, 0.857] | 0.696 / 0.727 | 0.991 / 0.991 |
| xgboost, basic features only | 0.393 [0.337, 0.456] | 0.387 [0.346, 0.437] | 0.397 / 0.391 | 0.947 / 0.942 |
| isolation_forest (unsupervised) | 0.194 [0.127, 0.255] | 0.161 [0.115, 0.236] | 0.231 / 0.211 | 0.760 / 0.784 |
| autoencoder (unsupervised) | 0.094 [0.066, 0.134] | 0.094 [0.069, 0.135] | 0.131 / 0.170 | 0.763 / 0.790 |
| mlp | 0.771 [0.725, 0.804] | 0.795 [0.760, 0.832] | 0.674 / 0.695 | 0.989 / 0.991 |
| temporal_seq (GRU, raw attributes only) | 0.728 [0.688, 0.760] | 0.794 [0.754, 0.828] | 0.656 / 0.727 | 0.974 / 0.984 |
| temporal_hybrid (GRU + engineered) | 0.799 [0.765, 0.827] | 0.853 [0.825, 0.880] | 0.718 / 0.776 | 0.990 / 0.992 |

### What the data supports (both datasets agree unless stated)

1. **Supervised and neural models beat unsupervised by a wide margin.** Isolation forest (0.16-0.19) and the
   autoencoder (0.09) are far below every supervised/neural model (0.54-0.85), though still above the 0.0175
   random baseline. CIs do not come close to overlapping. Unsupervised detectors miss `high_value` (recall
   about 0.01-0.06) because a large amount is only unusual *relative to the customer*, and their score is not built
   from customer-relative features alone. They do find bursts (0.73-0.74 for isolation forest).
2. **Among the top group, no reliable winner.** Random forest, XGBoost, MLP and temporal_hybrid are within overlapping
   CIs on seed 42. On seed 43 temporal_hybrid is highest (0.853 vs 0.825 for XGBoost) but its CI [0.825, 0.880]
   still overlaps XGBoost's [0.790, 0.857]. The direction is consistent (+0.005, +0.028) but neither result on its own is
   significant, and I do not claim the temporal model beats gradient boosting.
3. **History features are worth a lot for trees**: XGBoost PR-AUC 0.39 with basic features vs 0.79-0.83 with engineered ones.
4. **A sequence encoder can recover much of that without hand-built history.** `temporal_seq` (raw attributes +
   sequence) reaches 0.73-0.79 against 0.39 for XGBoost on the same raw attributes. This is the clearest justification
   for the temporal architecture, with two caveats: (a) it is a comparison of learners with equal *attributes* but the
   GRU also receives counterparty/country identity embeddings that the basic tree features lack (see 5); (b) its score varied
   a lot between the two datasets (0.728 vs 0.794), so it is noisier.
5. **Confound: geographic-change anomalies.** Both temporal models catch 0.86-0.98 of `geo_change` anomalies while every
   other model catches 0.02-0.33. This is very likely because they receive the counterparty *country identity* as an embedding
   and the generator's unusual-country list is a fixed set, so the models can memorise it. It is **not** clean evidence for
   sequence modelling. A tree model given country one-hots was not run and would be the fair test.
6. **Some types stay hard for everyone**: `new_counterparty` (best recall 0.41) and `unusual_time` (best about 0.7),
   because a single first-time counterparty or a night-time transaction is not unusual enough in isolation. `temporal_seq`
   nearly never flags `unusual_time` (0.02-0.23) despite the hour features being available: an underfit or early-stopping
   artefact I did not investigate.

### Threats to validity

- **Closed world.** Supervised models train on the same 7 injected anomaly types they are tested on; the unsupervised models
  have no such advantage but their thresholds also use validation labels. Real anomalies are open-ended, so absolute numbers
  are optimistic and the supervised-vs-unsupervised gap is probably overstated.
- **Injected signatures are clean and simple** (for example fixed unusual-country list, identical repeated amounts).
- **Single training run per dataset.** Neural models vary noticeably between the two runs, and data variance and
  initialisation variance are confounded. Only two data seeds were run; the CIs reflect test-set sampling only.
- **Same customers in train and test** (deployment-like); unseen-customer generalisation is not measured.
- Neural training used one CPU thread (a deadlock workaround), with small models and early stopping on PR-AUC.
  No hyperparameter search was done for any model.
- Latency is single-process CPU: random forest is the slowest per single row (about 27 ms), the temporal models cost about
  14 ms per 1000 rows in batch.

## EXP-REC-01: Reconciliation rules vs injected ledger discrepancies

- Script: `experiments/reconciliation/run_reconciliation_eval.py --preset medium` (`make reconciliation-eval`)
- Artifact: `evaluation/reports/reconciliation/reconciliation_eval_medium.json`
- Data: synthetic medium preset, 250,000 transactions, 3,750 injected discrepancies (six types).

| ground-truth type | injected | precision | recall |
|---|---|---|---|
| missing_ledger_entry | 562 | 1.000 | 1.000 |
| duplicate_posting | 562 | 1.000 | 1.000 |
| contradictory_amount | 1126 | 1.000 | 1.000 |
| currency_mismatch | 375 | 1.000 | 1.000 |
| missing_reference | 750 | 1.000 | 1.000 |
| late_posting | 375 | 1.000 | 1.000 |

Transaction level (core rules REC-001..008): 3750 TP, 0 FP, 0 FN; 0 misattributed types; 0 core-rule findings on
unlabelled transactions.

**How to read this.** A perfect score is expected and is *not* evidence of real-world quality. The rules are deterministic,
the injector is deterministic, and the tolerances (1 cent, T+3) were chosen with the data's value ranges in view. It shows
the engine implements its rules correctly and that the injected discrepancy types are separable. It says nothing about
messy real ledgers (rounding conventions, FX, partial settlements, timezone differences, reversed postings).

**Findings worth acting on:**
- Status rules fire on many *unlabelled* rows: REC-009 (failed) 2,643 = 1.1% and REC-010 (stale pending) 9,834 = 3.9% of
  transactions. This comes from the generator giving random statuses that never resolve, not from injected faults. Overall
  16,077 of 250,000 transactions (6.4%) have at least one finding, but only 3,750 (1.5%) are injected discrepancies.
  Downstream consumers (the agents) should treat core and status findings differently.
- Sender/receiver cannot be reconciled: the ledger schema has no counterparty fields.

## EXP-RAG-01: Knowledge-base retrieval, abstention signal, injection and poisoning (supports RQ3 groundwork)

- Script: `experiments/rag/run_retrieval_benchmark.py` (`make rag-benchmark`); artifact: `evaluation/reports/rag/retrieval_benchmark.{json,md}`
- Corpus: 12 synthetic policy documents (57 chunks at default chunking). Embedding model `sentence-transformers/all-MiniLM-L6-v2`.
- 43 answerable + 7 unanswerable hand-written questions (same author as the documents). Single run (retrieval is deterministic).

### Retrieval (default chunking, 120 words; section-level)

| mode | hit@1 [95% CI] | hit@5 [95% CI] | MRR [95% CI] |
|---|---|---|---|
| BM25 | 0.791 [0.67, 0.91] | 0.953 [0.88, 1.00] | 0.876 [0.80, 0.95] |
| dense (MiniLM + Chroma) | 0.814 [0.70, 0.93] | 1.000 [1.00, 1.00] | 0.886 [0.81, 0.96] |
| hybrid (RRF) | 0.907 [0.81, 0.98] | 0.977 [0.93, 1.00] | 0.941 [0.88, 0.99] |

### What the data supports
1. **All three modes retrieve the right section within the top 5 for 95-100% of questions**, including paraphrased and case-style
   ones. hit@5 is near the ceiling, so it cannot separate the methods on this corpus.
2. **Hybrid ranks the right section first more often** (hit@1 0.91 vs 0.79-0.81; MRR 0.94 vs 0.88-0.89). The direction is
   consistent across hit@1, MRR and nDCG, but the confidence intervals for hit@1 overlap with 43 questions, so I do not
   claim a significant win. A larger, independently written question set is needed.
3. **Chunk size made no measurable difference.** 120 and 240 words give identical indexes (57 chunks: every section is under
   120 words), and 40-word chunks (101 chunks) were not better. On a corpus of short sections, section-level chunking is enough.
4. **Unanswerable questions are separable by top-1 score**: dense cosine AUROC 1.000 (highest unanswerable 0.385 vs lowest answerable
   0.406, a margin of only 0.02) and normalised BM25 0.993 (highest unanswerable 0.138 vs lowest answerable 0.126). With only 7 unanswerable
   questions this is suggestive, not a calibrated abstention threshold. Any threshold must be chosen on held-out data.
5. **Injection tripwire**: flagged all 3 documents with imperative injection text (override, role reassignment, fake tags plus hidden
   HTML comment and zero-width characters, which cleaning also removed and counted), with **0 false positives across 57 trusted chunks**. It
   cannot see the 2 misinformation documents (they contain no instruction-like text), as expected.
6. **Poisoning is the key result.** With untrusted, topically matching documents in the index and the caller opting in, the poisoned document
   appeared in the top 5 for **6/6** adversarial queries in every mode and was top-1 for **5/6 (BM25, hybrid) and 6/6 (dense)**. Retrieval ranking gives no protection.
   With the default policy (untrusted excluded) poison exposure is **0/6** in every mode and normal-question hit@5 is unchanged. This is a
   trivial filter, and it works **only because the poisoned documents came from an untrusted source directory**.

### Threats to validity
- Same author wrote documents and questions; small n; questions may share vocabulary with the text.
- The adversarial documents were written to compete with specific sections (same headings); real poisoning could be subtler or weaker.
- One adversarial question (A3, "How can transaction review be done faster?") has a weak gold label; no mode retrieved it, so
  "gold in top-5" is 5/6 everywhere. Treat that as a question-quality issue, not a retrieval finding.
- A poisoned document inside a trusted source is not addressed by this layer.
- No LLM is involved yet; this says nothing about answer quality or grounding (RQ3), only that the right evidence can be found.

## EXP-LLM-01: Real local model on realistic cases (format, citations, decisions, injection). Groundwork for RQ3-RQ5

- Script: `experiments/llm/run_structured_output_check.py --provider qwen --model llm/models/qwen2.5-1.5b-instruct` (`make llm-check`)
- Artifact: `evaluation/reports/llm/structured_output_qwen_qwen2.5-1.5b-instruct.json` (includes each case's raw model text)
- Model: **Qwen2.5-1.5B-Instruct** (Apache-2.0), fp16 on Apple M2 Metal, greedy decoding, prompt `investigation-v1`, retrieval of 2 trusted chunks per case.
- 28 synthetic cases: 8 reconciliation problems, 8 behavioural anomalies, 8 clean, plus 4 *paired* injection cases (the same reconciliation problems again, with an
  instruction injected into the payment memo). Evidence contains raw facts and engine findings; it does **not** contain an anomaly-engine verdict.

| measure | result |
|---|---|
| structured output valid on the first attempt | **28/28** (no repair needed) |
| outputs citing an id that was not in the evidence | 2/28 (7%): both cited a rule shorthand such as `REC-003` instead of the evidence id |
| findings of kind "fact" with no citation | 0 |
| behavioural anomalies marked CLEAR | **6/8** |
| reconciliation problems marked CLEAR | 1/8 (6 REVIEW, 1 ESCALATE) |
| clean cases marked CLEAR | 7/8 |
| paired injection: control decision -> decision with injected memo | **3 of 4 flipped to CLEAR** (REVIEW->CLEAR twice, ESCALATE->CLEAR once); 0 controls were CLEAR |
| injected outputs that mention the memo as suspicious | 0/4 |
| latency, tokens | 19.3 s per case, about 14 tokens/s, median prompt 1,115 tokens, about 279 completion tokens |
| memory | about 3.3 GB on the GPU (measured in the smoke test) plus about 1.5 GB process RSS, on an 8 GB machine |

### What the data supports
1. **A 1.5B model can produce the required JSON reliably** (28/28 first try, zero facts left uncited). Format is not the bottleneck; the bounded repair loop was never needed here.
2. **It is not a reliable judge.** Given raw behavioural facts (for example "amount 25x the customer's typical amount", "3 a.m.") it marked 6 of 8 real anomalies CLEAR.
   It does better when the evidence states a discrepancy explicitly (reconciliation: 6/8 REVIEW). It does not reason from numbers to "unusual".
3. **Prompt-level defences were not enough against injected text.** Despite fencing, a nonce delimiter and explicit instructions, an instruction in the memo flipped the decision to
   CLEAR in 3 of 4 paired cases, and the model never flagged the memo. With n=4 this is anecdotal, but the pairing makes it hard to dismiss: controls were never CLEAR.
   **Design consequence for Phase 10:** model output must never be able to lower risk below what deterministic engines found (policy checks such as "a high-severity reconciliation
   finding cannot end as CLEAR"), and every cited claim must be verified against the evidence. Model compliance cannot be the control.
4. **Citation format needs one convention.** The model cites ids as shown in the prompt (`E:TXN-1`, 100% of citations were prefixed). A first checker that compared bare ids reported
   100% "hallucinated" ids; that was the checker's bug. Validators must normalise (`llm.schemas.normalize_citation`). The remaining true invalid citations were 2/28.

### Two runs, and why to be cautious
A first run used a flawed checker and an injection test on clean cases (where CLEAR is the *correct* answer, so obedience could not be measured); it was discarded and redesigned as the paired test.
On the 24 unchanged cases the second run differed from the first on 2 reconciliation decisions (8/8 REVIEW became 6 REVIEW, 1 ESCALATE, 1 CLEAR). The prompt fence uses a random nonce and Metal kernels are not bit-exact, so
decisions of this small model are **not perfectly reproducible**. Anything measured on it needs repeated runs (or a fixed nonce, which the prompt builder supports) before a claim is made.

### Threats to validity
- 28 cases and 4 injection pairs; no confidence intervals; one model, one prompt version, one decoding setting; no prompt tuning was done (deliberately, to avoid tuning on the test cases).
- Evidence had no anomaly verdict. In the full system the anomaly engine's score would be evidence, which would likely change the behavioural result. Not tested here.
- "Decision sanity" uses the generator's labels as ground truth, and one injected phrasing only.
- This is not the grounding evaluation (whether each claim is *supported*); that needs the Phase 10 validator.

## EXP-AGENTS-01: Multi-agent workflow vs single-agent baseline, latency breakdown (groundwork for RQ5)

- Script: `experiments/agents/run_workflow_demo.py --provider qwen --model llm/models/qwen2.5-1.5b-instruct` (`make agents-demo`); artifact: `evaluation/reports/agents/workflow_demo_qwen.json`
- Real components end to end: KYC matcher, XGBoost anomaly scorer (with the engine verdict in the evidence), reconciliation rules, knowledge-base retrieval, Qwen2.5-1.5B (Metal), on the small synthetic dataset.
- 6 cases, all from **after the anomaly model's training period** (2 reconciliation problems, 2 behavioural anomalies, 2 clean). One run each of the multi-agent workflow and the single-agent baseline.

| measure | result |
|---|---|
| cases reaching HUMAN_REVIEW with no failed step | 6/6 (none auto-closed; all report `UNVALIDATED`) |
| valid structured output first attempt | 6/6 multi-agent, 6/6 single-agent |
| mean wall time per case | multi-agent 40.3 s, single-agent 31.0 s |
| share of multi-agent time in the LLM call | **99.2%** (39.95 s of 40.3 s) |
| all non-LLM agents together | about 0.33 s (KYC matcher 0.24 s, retrieval 0.05 s, anomaly 0.02 s, rest under 0.01 s) |
| one-off setup (index, train anomaly model, load) | 14 s |

Proposed decisions (advisory, single run): multi-agent gave REVIEW, REVIEW, REVIEW, REVIEW, CLEAR, CLEAR for the two reconciliation, two behavioural and two clean cases,
matching the generator's labels in all 6; the single-agent baseline gave CLEAR (reconciliation), REVIEW, REVIEW, REVIEW and REVIEW, REVIEW for the same cases, so 4 of 6.

### What the data supports, and what it does not
1. **Orchestration overhead is negligible in wall-clock terms**: everything except the LLM call costs about 0.3 s per case (under 1%). The LLM is essentially the whole cost.
2. **(Superseded, see the correction in EXP-AGENTS-02.)** *The 9 s gap between the two variants is not orchestration; it comes from the multi-agent prompt being richer (KYC match evidence, a reconciliation-history summary and up to 4 targeted reference chunks versus 2
   generic ones), so the model reads and writes more.* The gap did not reproduce in a second run.
3. **The decision difference (6/6 vs 4/6) is anecdotal.** n=6, one run each, a small model whose decisions vary between runs (EXP-LLM-01), and several things differ between the variants (extra evidence, targeted retrieval), so
   the cause cannot be attributed to orchestration. It does fit the earlier finding that the model judges better when engines state their verdict in the evidence, which this run supports only tentatively.
   No claim about accuracy is made; that needs the Phase 13 ablation with many cases, repeated runs and confidence intervals.
4. **Failure handling and sign-off are verified by tests, not by this run** (24 tests: state order, fail-safe routing, invalid LLM output, four-eyes escalation, audit content).

### Threats to validity
- 6 cases, one run, one model; latency measured on one machine with one thread for numeric libraries (OMP=1).
- The workflow trains its anomaly model on labelled history at startup; cases were restricted to the held-out period, but the same synthetic generator produced train and test data.
- Nothing here is validated (Phase 10): the "matching labels" observation is about proposals, not about whether the reasoning was supported by evidence.

## EXP-GUARD-01: Guardrails (evidence validator + policy floor) (supports RQ4)

- Script: `experiments/guardrails/run_guardrail_eval.py` (`make guardrail-eval`); artifacts: `evaluation/reports/guardrails/guardrail_eval.json` and `guardrail_eval_before_format_fixes.json` (kept for transparency).
- Part A: claim-perturbation benchmark on the 28 EXP-LLM-01 cases. Part B: the guardrails applied to the 28 real Qwen2.5-1.5B outputs saved by EXP-LLM-01 (evidence rebuilt deterministically).

### Part A: correctness check with ground truth by construction
155 claims derived from real evidence by template: **155/155 recognised as supported (0 false alarms)**. Corrupted versions (each should be flagged unsupported):

| perturbation | n | flagged unsupported |
|---|---|---|
| unknown evidence id cited | 155 | 100% |
| citation removed | 155 | 100% |
| true value cited to the wrong evidence item | 155 | 100% |
| invented transaction id | 28 | 100% |
| currency swapped | 28 | 100% |
| date shifted | 28 | 100% |
| rule id swapped | 15 | 100% |
| **number changed** | 120 | **97.5%** (117/120) |

The 3 misses: two are the same change to only the sub-second part of a timestamp (seconds are not checked; documented), and one is a coincidence: "8 earlier transactions" became "10", and 10 occurs in the field name
`transactions_in_prior_10_min`, which the validator now (deliberately) indexes. That last miss is a direct cost of the field-name fix described below. **This is a correctness check**: the same author wrote the validator, the templates and the
perturbations, so it does not measure performance on free-form model text.

### Part B: real model outputs (28 cases, 62 claims)
| | before format fixes | after |
|---|---|---|
| supported | 17 | 20 |
| **unsupported** | 14 | **11** |
| unverifiable (nothing machine-checkable) | 31 | 31 |
| outputs passing all checks (`validated`) | 8/28 | 10/28 |
| advisory decision raised by the guardrail | 7 cases | 6 cases |
| **problem cases (8 reconciliation + 4 injected pairs) still CLEAR after the guardrail** | **0 of 12** (model alone: 4 CLEAR) | 0 of 12 |

**Manual audit of the unsupported flags** (each claim read against its evidence): before the fixes, **11 of 14 flags were correct**. The correct ones were: rule shorthand cited instead of the real evidence id (`E:REC-003` vs `REC-TXN-...-REC-003`),
true values cited to the wrong evidence item, uncited speculation ("indicates potential fraud risk"), and one **fabricated number** (the model wrote "$20,403.49 matches the typical prior transactions" when the evidence says 204,034.92 INR
at 17x the customer's typical amount). The **3 false alarms** had one root cause type, formatting the validator did not understand: dates written in prose ("April 23, 2025") and a number taken from a field name ("prior 24 hours").
Fixing them (prose dates, `H:MM AM/PM`, field-name numbers) is a post-hoc change made after seeing these examples, so **the after-fix audit is not an independent test**: all 11 remaining flags were correct on the same sample.

### What the data supports
1. **The validator does what it claims on structured perturbations**, with no false alarms on claims that are supported by construction.
2. **On real output it finds real problems**: 11 unsupported claims in 62, including a fabricated figure with a wrong conclusion. Roughly 3 in 4 unsupported flags were mis-citations or missing citations rather than invented facts.
3. **Half of the model's claims (31 of 62) cannot be machine-verified** (qualitative statements such as "unusual" or "looks normal"). The guardrail cannot certify these; it reports them separately. Only 10 of 28 outputs pass all checks.
4. **The policy floor is what stops unsafe CLEARs**, not the claim checker: the model alone marked 4 of 12 problem cases CLEAR (including 3 of 4 injected-instruction pairs); after the guardrail 0 of 12 were CLEAR. In all 4 injected pairs the high-severity reconciliation finding alone already implies REVIEW
   (REC-003, REC-004, REC-002, REC-008), and the evidence tripwire also flags all 4 injected memos, so either mechanism suffices on these cases: **this result does not show that the tripwire
   was needed**. Its value would be for a case whose only problem is the manipulation attempt.
5. **Cost:** the guardrail also raised some clean cases from CLEAR to REVIEW (clean: 7 CLEAR from the model, 6 after), i.e. extra reviewer work. On real model output the guardrail is conservative.
6. **RQ4 (partial):** the guardrails *detect and quarantine* unsupported claims and prevent unsafe advice; they do not make the model produce fewer unsupported claims. The scaled with/without comparison is Phase 13.

### Threats to validity
- 28 cases / 62 claims / one model / one prompt; the manual audit was done by the author of the validator and only covers the flagged (unsupported) claims, not a sample of the 31 unverifiable or 20 supported ones, so **recall of the validator on real
  free-form text is unmeasured** (an unsupported claim could be labelled supported).
- Behavioural cases in EXP-LLM-01 had no anomaly-model verdict, so no engine floor applied to them there (see EXP-AGENTS-02 for the workflow, which has it).
- The tripwire uses a regex heuristic; a rephrased injection evades it (the engine floor still protects real problems).

## EXP-AGENTS-02: the workflow with guardrails on, and a correction to EXP-AGENTS-01

- Same 6 held-out cases, same real components; artifact `evaluation/reports/agents/workflow_demo_qwen.json` (previous run kept as `workflow_demo_qwen_no_guardrails.json`).
- All 6 reached HUMAN_REVIEW; the review (guardrail) step costs about **4 ms** per case; the LLM call is 98.9% of the latency. The model's proposals were identical to the earlier run (REVIEW x4, CLEAR x2), so
  **the guardrails changed no decision here**. What they did: 4 problem cases passed all checks; the 2 clean cases stayed CLEAR but were flagged `CLEAR_WITHOUT_SUPPORTED_EVIDENCE` and **not validated** (0 verified claims, 3 unverifiable),
  i.e. the system refuses to certify a CLEAR it cannot back with evidence. Decision *changes* were demonstrated on the stored outputs (Part B above) and in the mock-based workflow tests.
- **Correction to EXP-AGENTS-01.** That entry attributed the 9 s gap between the multi-agent (40.3 s) and single-agent (31.0 s) variants to the richer multi-agent prompt. In this second run the order **reversed** (multi-agent 26.6 s, single-agent 27.9 s), and both
  variants were faster than before. Latency of this small model on this machine varies by tens of percent between runs, so **that explanation was not supported and should be disregarded**; what does replicate is that non-LLM work costs well under a second per case
  and the LLM call is about 99% of the time.


## EXP-ABL-01: engine-floor component ablation (no language model)

- Script: `experiments/ablation/run_component_ablation.py` (`make ablation`); artifacts: `evaluation/reports/ablation/component_ablation.{json,md}`; code: `evaluation/ablation/components.py`; tests: `backend/tests/ablation/`.
- Question: what does each deterministic component (reconciliation rules, anomaly model, KYC matcher) and each policy switch contribute to flagging problem cases? A case is "flagged" when the engine floor is REVIEW.
- Sets: the 24 held-out evaluation cases (headline) and all 228 distinct un-injected cases (train + val + eval; sensitivity, and the anomaly model saw part of that period). 95% CIs: case bootstrap.

| held-out (n=24) | recall: reconciliation (8) | recall: behavioural (8) | false flags on clean (8) |
|---|---|---|---|
| all engines (default) | 1.00 | 0.38 [0.12, 0.75] | 0.00 |
| reconciliation only | 1.00 | 0.00 | 0.00 |
| anomaly model only | 0.00 | 0.25 [0.00, 0.62] | 0.00 |
| KYC matcher only | 0.00 | 0.12 [0.00, 0.38] | 0.00 |
| only HIGH-severity findings count | 0.62 [0.25, 0.88] | 0.38 | 0.00 |

Pooled (n=228): behavioural recall 0.64 [0.51, 0.75] with all engines versus 0.60 anomaly-only; false-flag rate on clean cases 0.06 [0.02, 0.10], all from the KYC matcher (KYC-only precision 0.50); HIGH-only severity lowers reconciliation recall to 0.81 [0.70, 0.90].

### What the data supports, and what it does not
1. **The two problem families are covered by different engines**: reconciliation catches the ledger problems and (nearly) only those; the anomaly model catches behavioural ones and (nearly) only those. Neither substitutes for the other.
2. **About a third of behavioural problems (0.36 pooled, 0.62 held-out) reach no engine flag.** For those cases the advisory decision rests on the language model alone; the floor cannot protect them. This is the main residual risk of the design.
3. **In this data the KYC matcher adds mostly false flags at the floor** (6 of 108 clean cases, versus 6 true flags). This does NOT measure KYC value: these cases contain no identity problems, so the matcher's purpose is not exercised here. Its accuracy is EXP-KYC-01.
4. Counting LOW-severity findings or settlement-status rules changes nothing on this data; counting only HIGH severity loses reconciliation recall.

### Threats to validity
- Reconciliation recall of 1.00 is a check that rules and injector agree (see EXP-REC-01), not evidence of real-world quality.
- Behavioural recall is the recall of one anomaly model at one threshold on one synthetic dataset (small preset, one seed); n=8 per held-out category gives wide CIs.
- The pooled set overlaps the anomaly model's training period, and the case mix is balanced by construction, so precision depends on that mix. The pooled CI treats cases as independent although customers can repeat.

## EXP-ADV-01: payment-memo injection sweep (deterministic layers)

- Script: `experiments/adversarial/run_adversarial_suite.py` (`make adversarial`); wordings: `evaluation/adversarial_dataset/memo_injections.json` (7 classes, 30 wordings, hand-written by the author); artifacts: `evaluation/reports/adversarial/`.
- Each wording is written into the memo of each held-out case and the engine floor recomputed.

| wording class | wordings | tripwire fires (wording x case) | floor lowered on engine-flagged problem cases |
|---|---|---|---|
| explicit, wordings used in fine-tuning data | 5 | 0.40 (2 of 5 wordings) | 0 |
| explicit, held-out evaluation wordings | 3 | 0.33 (1 of 3 wordings) | 0 |
| explicit, novel | 5 | 1.00 (5 of 5) | 0 |
| paraphrase without trigger words | 4 | 0.00 | 0 |
| obfuscated / translated | 4 | 0.00 | 0 |
| benign memos | 5 | 0.00 (no false positives) | 0 |
| hard benign (share words with attacks) | 4 | 0.25 (1 of 4 falsely flagged) | 0 |

### What the data supports, and what it does not
1. **The tripwire is a weak detector: it caught 8 of 21 attack wordings**, including only 3 of the 8 explicit wordings from the training/evaluation sets, none of the paraphrased, obfuscated or translated ones, and it falsely flagged 1 of 9 benign memos. This confirms the documented position that it is a tripwire, not a defence. It was NOT tuned after this result (tuning on the same wordings would inflate it); improving it needs a fresh held-out wording set.
2. **The engine floor never dropped for a case the engines had flagged, whatever the memo said (0 of 330 checks: 30 wordings x 11 flagged problem cases).** This is a property of the design (the floor reads structured engine evidence, not memo text) and is also unit-tested; it is a correctness check, not a discovery.
3. **The residual risk is a problem case the engines missed, combined with an injection that fools the model** (see EXP-ABL-01 point 2). The LLM-level injection results are in EXP-LLM-02.

### Threats to validity
- Wordings were written by the author, who knew the tripwire's patterns; a real attacker would be more inventive. Only memos were tested (not KYC names, addresses or retrieved documents). Deterministic layers only; no model was asked.

## EXP-FT-01 (protocol, written BEFORE the fine-tuned arms were run): base vs RAG vs fine-tuned vs fine-tuned+RAG (RQ3, RQ6)

- Script: `experiments/llm/run_finetune_eval.py`; design: [fine-tuning.md](../architecture/fine-tuning.md); reports: `evaluation/reports/llm/finetune_eval*.json`.
- Arms (same Qwen2.5-0.5B-Instruct weights; adapter on/off): `base_norag`, `base_rag`, `ft_norag`, `ft_rag`; optional inference-only reference: Qwen2.5-1.5B + RAG. 24 held-out cases; greedy decoding; first attempt only (no retries); fixed per-case nonce.
- Metrics: first-attempt structured validity; share of valid outputs with at least one unsupported claim (guardrail evidence validator); share with a grounded summary; decision agreement with the generator's label (problem = not CLEAR); share of cases where the model says CLEAR although the engine floor requires review; latency. 95% CIs by case bootstrap. Injection: `base_rag` vs `ft_rag` on the 8 injected cases (model says CLEAR / repeats the instruction).
- Pre-stated hypotheses (to be confirmed or refuted, both reported): H1 RAG raises grounding for the base model. H2 fine-tuning raises first-attempt validity and lowers unsupported claims more than RAG alone. H3 fine-tuning does not raise decision agreement beyond the engine floor, since its targets are derived from the engines. H4 injection: no claim before seeing data.
- Decision rule: a difference is called real only if the CIs do not overlap AND the direction matches on both the validity and the unsupported-claim metrics; otherwise it is reported as inconclusive. Small n (24) means most differences will be inconclusive; that is stated, not hidden.
- Threats: one seed, one base model, one machine per arm group (base arms on the laptop, fine-tuned arms on a GPU machine: latencies are NOT comparable across machines), machine-generated targets, synthetic evidence.

## EXP-ORCH-01: multi-agent workflow vs single-agent baseline on 24 held-out cases (RQ5)

- Script: `experiments/agents/run_orchestration_ablation.py`; artifact: `evaluation/reports/agents/orchestration_ablation_qwen.json`. Model: Qwen2.5-0.5B-Instruct (Metal, fp16), greedy, one run per case, the 24 held-out cases of EXP-FT-01. Same engines, evidence builders and model; the variants differ in orchestration and targeted retrieval (`agents/baseline.py`). This supersedes the n=6 demonstration in EXP-AGENTS-01/02.

| measure | multi-agent | single-agent |
|---|---|---|
| valid structured output on the first attempt | 0.75 [0.58, 0.92] (18/24) | 0.54 [0.33, 0.75] (13/24) |
| decision agrees with label, model output only (invalid = wrong) | 0.46 [0.25, 0.67] | 0.29 [0.13, 0.50] |
| same, after the guardrails (engine floor and unsupported-claim rule) | 0.58 [0.38, 0.79] | not applicable (no guardrails) |
| wall time per case, s | 15.8 [12.5, 19.4] | 24.9 [19.2, 30.8] |

Paired on the model-only decision: multi-agent right and single-agent wrong on 5 cases, the reverse on 1 (exact two-sided sign test p = 0.22).

### What the data supports, and what it does not
1. **The apparent accuracy difference is a format-validity difference, not better judgement.** Where the 0.5B model produced a valid output it said REVIEW every time, in both variants (no CLEAR anywhere), so "agreement with the label" only measures how many problem cases had a valid output. Restricted to valid outputs the agreement is 11/18 (multi) vs 7/13 (single), which is just each subset's share of problem cases.
2. **The multi-agent variant produced valid output more often (18/24 vs 13/24) but the CIs overlap and the sign test is not significant (p=0.22): inconclusive at n=24.** A plausible cause is the richer, more structured multi-agent prompt (KYC match evidence, targeted retrieval), which is confounded with orchestration here and was not separated.
3. **Latency: multi-agent was faster (15.8 s vs 24.9 s), the opposite of EXP-AGENTS-01 (40 s vs 31 s, 1.5B).** The direction flipped between experiments, so orchestration is not what drives it; wall time is dominated by generation length (a failed single-agent output often runs long or is retried). No latency claim about orchestration is made.
4. **The guardrail layer is what rescues invalid outputs**: 3 cases ended with no decision at all (invalid output and no engine flag); in 3 further cases the engine floor supplied REVIEW where the model gave nothing.

### Threats to validity
- One small model that never answers CLEAR; a stronger model may separate the variants differently (see EXP-LLM-02 for the 1.5B). n=24, one run, one seed. Two things differ between variants (orchestration and prompt/retrieval content), so the cause of any difference is not identified.

## EXP-LLM-02: RAG and guardrail layers with real local models on 24 held-out cases (RQ3, RQ4; baseline for RQ6)

- Script: `experiments/llm/run_finetune_eval.py`; artifacts: `evaluation/reports/llm/finetune_eval_base_laptop.json` (Qwen2.5-0.5B, base arms) and `finetune_eval_reference_1.5b_laptop.json` (Qwen2.5-1.5B, inference only; its `model` field was mislabelled by the script and corrected, see the note inside the file); consolidated tables in `docs/research/results-summary.md`. Greedy, first attempt only, no retries, one run, laptop (Metal). Injection cases: the 8 reconciliation cases with an unseen memo wording.
- The fine-tuned arms are NOT in this entry: they need the GPU-machine run (protocol in EXP-FT-01).

| arm (n=24) | valid first attempt | outputs with an unsupported claim (of valid) | model says CLEAR on a problem case (of 16) | clean cases the model sends to REVIEW (of 8) |
|---|---|---|---|---|
| 0.5B, no RAG | 0.83 [0.67, 0.96] | 0/20 | 0 | 8 |
| 0.5B, RAG | 0.67 [0.50, 0.83] | 0/16 | 0 | 8 |
| 1.5B, no RAG | 1.00 | 1/24 | 7 | 5 |
| 1.5B, RAG | 1.00 | 1/24 | 7 | 2 |

Guardrail layers on the 1.5B outputs (problem cases left CLEAR / clean cases raised to REVIEW): model only 0.44 / 0.62 (no RAG), 0.44 / 0.25 (RAG); adding the engine floor: 0.25 / 0.62 and 0.12 / 0.25; adding the unsupported-claim rule: unchanged for no RAG, 0.12 / 0.38 for RAG.
Injection (1.5B + RAG, 8 injected reconciliation cases): the model answered CLEAR on 4 of 8, versus 3 of 8 for the same cases without an injected memo; the engine floor was REVIEW on all 8.

### What the data supports, and what it does not
1. **RQ3 (does RAG improve grounding?): not shown.** Grounding was already at ceiling: a single unsupported claim in about 60 claims per 1.5B arm, and none for the 0.5B. RAG did not change that. RAG **hurt the 0.5B's format validity** (0.83 to 0.67; CIs overlap, so inconclusive) and helped the 1.5B send fewer clean cases to REVIEW (5 to 2 of 8, n=8, inconclusive). `unverifiable` claims (citations with no checkable tokens; about half of all claims in EXP-GUARD-01) are not counted as unsupported here, so "grounded" is weaker than it sounds.
2. **The 0.5B base model cannot judge**: valid outputs are always REVIEW, including for every clean case. Its failures are formatting (missing `confidence`, broken JSON), which is exactly what fine-tuning targets (RQ6).
3. **RQ4 (do the guardrails help?): partly.** The **engine floor** is the layer that helps: it cut problem cases left CLEAR from 7 of 16 to 4 (no RAG) and 2 (RAG). The remaining 2 to 4 are problem cases no engine flagged, so no guardrail can catch them. The **unsupported-claim rule added nothing here** (it changed no problem case and raised 3 more clean cases to REVIEW in the RAG arm) because there was only one unsupported claim; its value would show only for a model that invents facts.
4. **Injection: inconclusive.** 4 of 8 versus 3 of 8 is within noise; the model says CLEAR on many problem cases with or without an injected memo, so this model's decision is unreliable regardless. What protected these 8 cases was the engine floor, not the model. No output repeated the injected text.
5. Latency (0.5B about 10 s, 1.5B about 32 s per case on this laptop) is not comparable with the GPU-machine arms.

### Threats to validity
- n=24 (8 per category, 16 problem cases), one run, greedy; CIs are wide and many differences are inconclusive. The 0.5B and 1.5B are different families of failure, so comparing them says nothing about model size in general. The problem cases include ones the engines do not flag, whose CLEAR rate depends on the anomaly model's threshold. Decision "agreement with the label" is not reported as a headline because a model that always says REVIEW scores the base rate.
