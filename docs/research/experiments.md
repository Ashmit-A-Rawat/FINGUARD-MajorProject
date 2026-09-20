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
