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
