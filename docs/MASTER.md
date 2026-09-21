# FIN-GUARD: master document

The complete record of the project: what was built, every problem hit and how it was solved, every test and experiment with its result, every training run, what is left, and how to run everything.

> All data is **SYNTHETIC**. No real people, accounts or transactions. Results do not describe real-bank performance.
> Shorter companions: [PROGRESS.md](PROGRESS.md) (running summary), [research/experiments.md](research/experiments.md) (full experiment write-ups with threats to validity), [research/results-summary.md](research/results-summary.md) (generated tables), [research/reproduce.md](research/reproduce.md).

**Contents:** 1 What it is · 2 Chronological build log · 3 Every issue and its fix · 4 Every test and result · 5 Every training run · 6 Corrections and negative results · 7 What is left · 8 How to run the project · 9 Repository map

---

## 1. What FIN-GUARD is

An on-premise, multi-agent AI system for **KYC verification** and **transaction anomaly detection and remediation**, built as a research prototype.

Pipeline: synthetic data → validated consolidation → KYC engine + anomaly engine → reconciliation engine → knowledge base / RAG → local language model → agents → guardrails (evidence validation, policy floor, self-critique) → human sign-off → report + tamper-evident audit trail, with a FastAPI backend and a React UI.

Design rules that hold everywhere:
- Deterministic engines produce the evidence; the language model is **advisory and untrusted**.
- Final decision = max(model, engine floor). The model can never lower risk below what the engines found, and the floor never forces ESCALATE (a human judgement).
- Every model claim is checked against the case evidence (supported / unsupported / unverifiable).
- Knowledge-base trust is set by the source *directory*, not the file content; untrusted chunks are excluded by default.
- The audit trail is hash-chained per case with database-assigned sequence numbers; escalation needs a second, authenticated reviewer (four-eyes).
- Mock outputs are always labelled mock. No result is reported that was not produced by the code.

Environment used: Apple M2, 8 GB unified memory, macOS (Darwin 25), Python 3.11, no CUDA. Git identity: Ashmit-A-Rawat. Repository: <https://github.com/Ashmit-A-Rawat/FINGUARD-MajorProject> (20 commits, fast-forward pushes only).

---

## 2. Chronological build log

| # | Phase | Commit | What was built |
|---|---|---|---|
| 1 | Skeleton | `353bd07` | Repo layout, config (pydantic-settings, `.env.example`), FastAPI health endpoint, Postgres-only docker-compose, Makefile, ruff + mypy + pytest setup, ADR 0001 (Python 3.11, heavy dependencies as optional extras) |
| 2 | Synthetic data | `68fd371` | Domain schemas (every record `is_synthetic=True`, schema refuses `False`); deterministic generator (one `SeedSequence` per stage, fixed window ending 2025-06-30); presets small (1,000 customers / 20,000 tx), medium (10,000 / 250,000), large (100,000 / 1M, needs `--allow-large`); customers, KYC records with controlled variations (typos, abbreviations, transliteration, name-order swaps, address mismatch, DOB conflict, duplicates, namesakes), transactions with 7 injected behavioural anomaly types (2%), ledger with 6 injected discrepancy types; labels in **separate files** so they cannot leak into features; manifest with per-file SHA-256 |
| 3 | Data pipeline | `d4a88d3` | Validation + consolidation into a `ConsolidatedStore` and canonical case objects; invalid rows go to a quarantine (the system refuses to start if any row is quarantined) |
| 4 | KYC entity resolution | `7a446aa` | Stages: exact, fuzzy (rapidfuzz), own BM25 (score divided by query self-score) + char-trigram variant, dense (MiniLM + cosine), hybrid (alpha 0.5, fixed a priori), learned name reranker (HistGradientBoosting on pairwise name features, name-only), structured DOB/address verification with explicit contradiction strings; benchmark with entity-disjoint train/dev/test split and query-bootstrap CIs |
| 5 | Anomaly detection | `1a0dcf9` | 34 engineered causal features (16 "basic"); chronological 60/20/20 split with episode purge; 9 models (logistic regression, random forest, XGBoost, XGBoost basic features, isolation forest, autoencoder, MLP, GRU sequence model, GRU + engineered hybrid); customer-clustered bootstrap CIs; ADR 0002 (single-thread OpenMP) |
| 6 | Reconciliation | `64765a9` | Transaction-vs-ledger rules REC-001..REC-010 (core rules 001-008 map to the six injected types; 009/010 are settlement-status observations), severity levels, tolerances (1 cent, T+3) |
| 7 | Knowledge base | `080ccf1` | 12 synthetic policy documents (57 chunks), loader with cleaning (hidden HTML comments, zero-width characters, counted), chunker, BM25 + dense (Chroma) + hybrid (RRF) retrieval, manifest that rejects a different embedder, trust by source directory, regex injection tripwire; 50-question golden set + adversarial poisoning set |
| 8 | LLM layer | `b072381`, `e5a782f` | Provider interface, mock provider (always labelled), local Hugging Face provider (Qwen / Mistral), bounded repair loop for structured output, prompt `investigation-v1` with fenced, nonce-delimited evidence, `E:<id>` citation convention with normalisation, hardware assessment script, evidence builders, resumable weight downloader |
| 9 | Multi-agent workflow | `851602c` | Explicit 10-state machine (ADR 0003, not LangGraph): CASE_CREATED → DATA_READY → KYC_ANALYZED → ANOMALY_ANALYZED → RECONCILED → EVIDENCE_RETRIEVED → INVESTIGATION_GENERATED → SELF_CRITIQUED → HUMAN_REVIEW → CLOSED (only by a named human). Auditor, reconciliation, investigator, reviewer, report agents; typed contracts; audit events; fail-safe routing to HUMAN_REVIEW on any step failure; single-agent baseline |
| 10 | Guardrails | `2458bdd`, `0f1fc41` | Evidence validator (numbers, ids, dates, currencies, rule ids checked against cited evidence), engine policy floor, evidence injection tripwire, self-critique that never asks an LLM |
| 11 | API + UI + audit | `b532c7e` | FastAPI with scrypt password hashes, JWT with role re-read from the database each request, login throttling, roles (analyst / auditor / admin, separation of duties), hash-chained append-only audit trail with integrity check, four-eyes escalation; React 19 + Vite + TypeScript + Tailwind UI with the ten panels (inbox, customer/KYC, timeline, anomaly, reconciliation, evidence, AI investigation, self-critique and decision, audit trail); advisory banner, mock labels, uncalibrated-confidence label |
| 12 | Fine-tuning (code) | `36b273b`, `6724455` | Teacher targets from evidence only, dataset builder, LoRA trainer, adapter-aware provider, four-arm evaluation script, committed dataset (`data/finetune/`); **training itself was not completed** (see 5.5) |
| 13 | Evaluation + ablations | `b6a1768` | Engine-floor component ablation, memo-injection sweep, multi-agent vs single-agent evaluation, base and 1.5B LLM arms, guardrail-layer analysis, generated results summary |
| 14 | Deployment + docs | `b6a1768`, `59565d2` | Dockerfile (API), frontend Dockerfile + nginx with strict CSP, docker-compose (Postgres + API + web), CI workflow, reproduction guide, CITATION.cff |
| + | Tripwire improvement | `9665486` | Normalised regex, semantic (embedding) tripwire, four held-out wording sets, Colab notebook, dtype-safe trainer |

Housekeeping done along the way (on request): removed unused packages and placeholders (`backend/app/models`, `data_pipeline/entity_resolution`, `scripts/download_model.py`, `.gitkeep` files, empty `data/{raw,kyc,transactions,reconciliation}`, `notebooks/` at that time), deleted regenerable data and caches, created and maintain [PROGRESS.md](PROGRESS.md), and saved working preferences (no Claude attribution in commits, pause after each phase until "go", keep the short progress doc, prune unneeded files).

---

## 3. Every issue faced and how it was resolved

### 3.1 Environment and platform (macOS, Apple Silicon)
| Issue | Cause | Resolution |
|---|---|---|
| Segfaults (exit 139), model-load hangs, silent deadlocks at 0% CPU | Two OpenMP runtimes (PyTorch, XGBoost, scikit-learn, Chroma, sentence-transformers each bundle one) fighting in one process; import-order tricks and `KMP_DUPLICATE_LIB_OK` were insufficient | One process-wide `OMP_NUM_THREADS=1` in `backend/app/core/runtime.py` (ADR 0002), applied by the `kyc`, `anomaly_detection`, `knowledge_base` packages and the test conftest; verified with every library combined. Cost: no intra-op parallelism |
| `ModuleNotFoundError: No module named 'backend'` | macOS marks files in dot-directories hidden; Python ignores hidden `.pth` files, so the editable install silently stops working | Makefile sets `PYTHONPATH=.`; alternative `chflags -R nohidden .venv`; documented in the README |
| A helper script named `bisect.py` broke imports | Shadowed the standard library module | Renamed |
| `timeout` command missing, `sleep` chains blocked, zsh does not word-split variables | macOS / shell differences | Background until-loops instead of sleeps; quoted variables |
| Hugging Face client downloads stalled on this network | Network | Resumable `curl` downloader `scripts/fetch_weights.sh` that reconnects on stall; the unused `download_model.py` was later deleted |

### 3.2 Correctness bugs found during development
| Issue | Resolution |
|---|---|
| A shared in-memory Chroma collection name let one index destroy another | Per-index unique collection name plus a regression test |
| Audit-trail sequence race: background job, analyst view and sign-off interleaving dropped or duplicated events | Sequence numbers are assigned by the database at append time, not by the caller |
| `is_anomaly` parsed as a boolean so `== "True"` matched nothing | Compared as strings via `.astype(str).eq("True")` |
| A month/day bug in the validator's missing-token check (`_Missing.any()`) | Fixed with a regression test |
| macOS segfault importing PyTorch before XGBoost | XGBoost imported first in `anomaly_detection/__init__.py` and the test conftest; later replaced by the single-thread rule |

### 3.3 Repository and tooling problems
| Issue | Resolution |
|---|---|
| `hardware.py` was git-ignored by the rule `llm/models/*` | Rule changed to `llm/models/*/` and the file committed |
| **`knowledge_base/vector_store/chroma_store.py` was never committed** (rule `knowledge_base/vector_store/*`), so a fresh clone would have broken the knowledge base | Rule now ignores only index subdirectories; file tracked; it then showed 4 lint errors that had been hidden, which were fixed; `git ls-files --others --ignored` confirmed no other source file was hidden |
| Scripted text patches silently applied nothing after ruff reformatted the files, leaving edits missing | Patches now assert that the target text exists; edits re-viewed |
| A `sed` command failed on BSD sed and a chained Python step never ran, so a progress-doc update was skipped even though the commit went through | Detected from the output; update redone in Python and committed |
| A test directory named `evaluation/` shadowed the real `evaluation` package | Renamed to `backend/tests/ablation/`; no `__init__.py` in test directories |
| Harness reminder asked for a Claude co-author trailer on commits | Not added; the user's instruction (no Claude attribution) takes precedence; verified with `git log | grep -ci claude` = 0 |
| Ruff started linting the notebook | `notebooks` added to ruff `extend-exclude` |
| mypy errors (untyped lambda, Literal comparison overlap, Any return) | Replaced with a typed helper / `.value` comparison / annotation |

### 3.4 Wrong claims that were corrected in the record
- **EXP-AGENTS-01 latency explanation retracted.** It blamed the 9 s gap between multi-agent and single-agent on a richer prompt; a second run reversed the order (26.6 s vs 27.9 s), so the explanation was unsupported and is marked superseded.
- **EXP-GUARD-01 injected-pair sentence corrected** in its own commit (`0f1fc41`).
- **EXP-LLM-01 first run discarded**: its checker compared bare ids and reported 100% "hallucinated" ids (the model cites `E:<id>`, so the checker was wrong), and its injection test used clean cases where CLEAR is the correct answer, so obedience could not be measured. Redesigned as a paired test (the same problem cases with and without an injected memo) and a normalising citation checker.
- **Guardrail false alarms**: the validator flagged 3 of 14 unsupported claims wrongly (dates written in prose, `H:MM AM/PM`, a number taken from a field name). Fixed after seeing them, so the after-fix audit is explicitly *not* an independent test.
- A report's `model` field was mislabelled by the script for the 1.5B reference run; corrected in the JSON with a note inside the file.
- An RQ5 "decision accuracy" number initially conflated invalid outputs with wrong decisions; the write-up states that and adds the valid-only figures.

### 3.5 Model and hardware limits
| Issue | Resolution |
|---|---|
| Downloaded 1.5B model, then a LoRA probe on it thrashed memory (8 GB machine) | Asked the user; they chose fine-tuning the 0.5B with the 1.5B as an inference-only reference arm |
| 0.5B LoRA probe hung / never finished; a diagnostic showed every training sequence is 1,806-2,342 tokens (median 1,970) and one forward pass with loss ran the GPU out of memory (9.06 GiB allocated, limit 9.07 GiB) | Concluded training does not fit an 8 GB laptop. Everything else was built and wired, a guide for a GPU PC and a Colab notebook were written, and the trainer now picks float32 on GPUs without bf16 (T4). **Training has not been run** |
| Docker daemon not running on the laptop | Compose syntax and CI YAML validated only; images not built (see 7) |

### 3.6 Tripwire (prompt-injection detector) findings
1. The original regex tripwire caught 8 of 21 attack wordings and 0 of 24 on a fresh set.
2. Tuning the regex to 100% on the tuning sets was mostly overfitting: on wordings written afterwards it caught 7/30 (v3) and 6/24 (v4).
3. Broader patterns caused two real false positives (a trusted document that *describes* the attack: "to ignore rules"; a misinformation document that is a claim, not an instruction: "never need review"); the guard tests caught both and the patterns were tightened rather than the tests loosened.
4. The semantic (embedding) classifier is the real gain; a peek at several classifier settings on v3 made v3 a development set, so a fresh v4 set was written and scored once (result in 4.4, EXP-ADV-02).

---

## 4. Every test and its result

### 4.1 Automated tests
**Final state: 407 backend tests pass (about 23 s), 33 frontend tests pass, ruff clean, ruff format clean, mypy strict clean on 228 source files.** (Growth: 365 backend + 33 frontend at the end of phase 11; 380 after the phase-12 groundwork; 388 after phases 13-14; 407 now.)

Test functions per area (these counts include the newest files; parametrised cases raise the collected total to 407):

| Area | Functions | What they check (highlights) |
|---|---|---|
| data_foundation | 24 | schemas refuse `is_synthetic=False`; generator is bit-for-bit deterministic for a seed; labels are never columns of feature tables; manifest counts and hashes |
| data_pipeline | 23 | validation rules, quarantine, consolidation, canonical case building |
| kyc | 33 | each stage, metric definitions, bootstrap, structured DOB/address scoring, contradiction strings |
| anomaly | 22 | **feature causality by truncation invariance**, window counts against brute force, chronological split, episode purge, unsupervised models ignore labels (flip all labels: identical scores) |
| reconciliation | 23 | tolerance boundaries (1 cent, T+3), duplicate ordering, float noise, currency + amount together, immutability |
| knowledge_base | 37 | cleaning, chunking, retrieval modes, trust exclusion, manifest mismatch, poisoning, **no false injection flags on the 12 production documents**, tripwire normalisation and patterns, semantic tripwire with a fake embedder |
| llm | 28 | mock provider labelling, structured-output repair loop, citation normalisation, prompt fencing, adapter on/off switch (fake network) |
| guardrails | 27 | evidence validator per perturbation type, policy floor, CLEAR-below-floor, unsupported-claim routing |
| agents | 30 | state order, fail-safe routing, invalid LLM output, four-eyes escalation, audit content; teacher targets have 0 unsupported claims and grounded summaries across 60 real cases, never ESCALATE, deterministic |
| api | 31 | auth, throttling, role checks, personal-data visibility, audit chain integrity and tamper detection, no route name that freezes/blocks/reverses/contacts/files |
| ablation | 5 | engine-floor configurations, evaluation metrics, injection never lowers a flag |
| health | 2 | health endpoint |
| frontend (vitest) | 33 | API client, decision rules (CLEAR against advisory needs acknowledgement), components |

Additional checks done by hand: browser screenshots of the UI during phase 11 (puppeteer), an end-to-end API script with a real SQLite database, and manual reading of every "unsupported" claim flag against its evidence (EXP-GUARD-01).

### 4.2 Experiment results (headline numbers; full tables and caveats in [experiments.md](research/experiments.md))

**EXP-KYC-01 (RQ1)**: KYC matching, 4,000 customers, seeds 42 and 43 (independent replication, no code change).

| system | F1 seed 42 [95% CI] | F1 seed 43 [95% CI] |
|---|---|---|
| exact | 0.631 [0.605, 0.658] | 0.640 [0.615, 0.662] |
| fuzzy | 0.645 [0.622, 0.665] | 0.659 [0.640, 0.680] |
| BM25 word / char / dense / hybrid | 0.558 / 0.558 / 0.558 / 0.560 | 0.577 / 0.574 / 0.581 / 0.580 |
| hybrid + reranker | 0.714 [0.693, 0.735] | 0.735 [0.715, 0.757] |
| hybrid + structured | 0.971 [0.964, 0.977] | 0.977 [0.970, 0.983] |
| full (hybrid + reranker + structured) | 0.963 [0.955, 0.971] | 0.959 [0.951, 0.967] |

Reading: **the hybrid BM25 + dense did not beat its parts (RQ1 not supported for this design)**; the learned reranker helped on F1 but raised false matches; structured DOB/address verification made the biggest difference (0.71 → 0.97) and the reranker added nothing on top of it. Caveats: DOB is uniformly random and the name pool tiny, so structured verification is unrealistically decisive.

**EXP-ANOM-01 (RQ2)**: 250,000 transactions, 2% anomalies, chronological split, seeds 42 and 43 (independent datasets). PR-AUC [95% customer-clustered CI]:

| model | seed 42 | seed 43 |
|---|---|---|
| logistic regression | 0.537 [0.459, 0.594] | 0.572 [0.506, 0.637] |
| random forest | 0.804 [0.765, 0.831] | 0.814 [0.780, 0.844] |
| XGBoost | 0.794 [0.756, 0.826] | 0.825 [0.790, 0.857] |
| XGBoost, basic features | 0.393 [0.337, 0.456] | 0.387 [0.346, 0.437] |
| isolation forest (unsupervised) | 0.194 [0.127, 0.255] | 0.161 [0.115, 0.236] |
| autoencoder (unsupervised) | 0.094 [0.066, 0.134] | 0.094 [0.069, 0.135] |
| MLP | 0.771 [0.725, 0.804] | 0.795 [0.760, 0.832] |
| GRU sequence, raw attributes | 0.728 [0.688, 0.760] | 0.794 [0.754, 0.828] |
| GRU + engineered (hybrid) | 0.799 [0.765, 0.827] | 0.853 [0.825, 0.880] |

Reading: supervised and neural models beat unsupervised by a wide margin; **no reliable winner** among the top group (overlapping CIs); engineered history features roughly double tree-model PR-AUC; the temporal models' large advantage on geographic-change anomalies is likely a confound (they receive country identity embeddings); the random baseline is 0.0175.

**EXP-REC-01**: reconciliation vs injected ledger discrepancies: precision and recall 1.000 on all six types (3,750 TP, 0 FP, 0 FN). This is a check that rules and injector agree, **not** real-world quality. Status rules REC-009/010 fire on 1.1% / 3.9% of rows that carry no injected fault (generator statuses that never resolve).

**EXP-RAG-01**: 12 documents, 43 answerable + 7 unanswerable questions. hit@5: BM25 0.953, dense 1.000, hybrid 0.977; hit@1: 0.791 / 0.814 / 0.907 (CIs overlap); chunk size made no difference; unanswerable questions separable by top-1 score (dense AUROC 1.000, only 7 questions). Injection tripwire flagged all 3 instruction documents with 0 false positives on 57 trusted chunks. **Poisoning: with untrusted documents in the index the poisoned document reached the top 5 for 6/6 queries (top-1 for 5-6 of 6); the default trusted-only policy cut exposure to 0/6.**

**EXP-LLM-01**: Qwen2.5-1.5B, 28 cases: structured output valid first try 28/28; 2/28 outputs cited a non-existent id; model marked 6/8 behavioural anomalies CLEAR (no anomaly verdict in that evidence) and 1/8 reconciliation problems; **an injected memo flipped 3 of 4 paired cases to CLEAR** (controls: 0); 19.3 s per case (about 14 tokens/s); about 3.3 GB GPU memory. Decisions varied between two runs of the same cases (2 differing), so single runs are not reproducible.

**EXP-AGENTS-01/02**: 6 held-out cases with the real 1.5B model: all reached HUMAN_REVIEW; non-LLM work about 0.3 s per case, the LLM call about 99% of latency; guardrail review step about 4 ms; the guardrails changed no decision in that run (2 clean CLEARs were flagged `CLEAR_WITHOUT_SUPPORTED_EVIDENCE` and left unvalidated). Latency comparison retracted (3.4).

**EXP-GUARD-01**: Part A (claim perturbation, ground truth by construction): 155/155 supported claims recognised (0 false alarms); every perturbation type flagged 100% except number-changed 97.5% (117/120). Part B (28 real outputs, 62 claims): 20 supported, 11 unsupported, 31 unverifiable; only 10/28 outputs fully validated; **problem cases still CLEAR after the guardrail: 0 of 12 (the model alone had 4)**; the policy floor, not the claim checker, prevented the unsafe CLEARs; cost: one clean case raised to REVIEW.

**EXP-ABL-01** (engine-floor ablation, 24 held-out and 228 pooled cases; details in experiments.md): reconciliation and anomaly engines cover *different* problem families; held-out recall with all engines is 1.00 (reconciliation) and 0.38 [0.12, 0.75] (behavioural), 0 false flags on 8 clean; pooled behavioural recall 0.64 [0.51, 0.75]; **about a third of behavioural problems reach no engine flag (main residual risk)**; the KYC matcher adds mostly false flags here because these cases contain no identity problems; counting only HIGH-severity findings drops reconciliation recall to 0.62 (held-out).

**EXP-ADV-01** (memo-injection sweep, 30 wordings × held-out cases): **the engine floor was never lowered by any memo (0 of 330 checks)**, a design property that is also unit-tested. The *original* tripwire caught only 8 of 21 attack wordings (report kept as `memo_injection_sweep_original_regex.*`).

**EXP-LLM-02** (24 held-out cases, first attempt, greedy):

| arm | valid first attempt | outputs with an unsupported claim | model says CLEAR on a problem case (of 16) |
|---|---|---|---|
| 0.5B, no RAG | 0.83 [0.67, 0.96] | 0/20 | 0 |
| 0.5B, RAG | 0.67 [0.50, 0.83] | 0/16 | 0 |
| 1.5B, no RAG | 1.00 | 1/24 | 7 |
| 1.5B, RAG | 1.00 | 1/24 | 7 |

The 0.5B failures are formatting (missing `confidence`, broken JSON), and its valid outputs are always REVIEW, even on every clean case. Guardrail layers on the 1.5B: problem cases left CLEAR 0.44 (model only) → 0.25 (no RAG) / 0.12 (RAG) with the engine floor; the unsupported-claim rule changed no problem case and raised 3 more clean cases in the RAG arm (only one unsupported claim existed). Injection (1.5B + RAG, 8 cases): CLEAR on 4/8 vs 3/8 without the memo, inconclusive; the floor was REVIEW on all 8. **RQ3: not shown** (grounding already at ceiling). **RQ4: the engine floor helps; the claim rule added nothing here.**

**EXP-ORCH-01 (RQ5)**, 0.5B, 24 cases: valid first attempt multi 0.75 [0.58, 0.92] vs single 0.54 [0.33, 0.75]; decision agreement 0.46 vs 0.29 (mostly a validity effect: valid outputs were always REVIEW); paired 5 vs 1, sign test p = 0.22 → **inconclusive**; wall time 15.8 s vs 24.9 s, the *reverse* of EXP-AGENTS-01 (40.3 vs 31.0 s), so no latency claim about orchestration.

**EXP-ADV-02 (tripwire improvement)**, blind set v4 (24 attacks, 20 benign memos, written after the classifier configuration was fixed, scored once):

| detector | recall | false positives |
|---|---|---|
| improved regex | 0.25 [0.12, 0.45] (6/24) | 0/20 |
| **semantic classifier** | **0.79 [0.60, 0.91] (19/24)** | 0/20 (CI upper bound 0.16) |
| either | 0.79 | 0/20 |

Other sets: v1 (dev) regex 8/21 → 21/21 after tuning (tuned on it); v2 (seen while designing) 0/24 → 24/24; v3 regex 7/30; semantic on v3 28/30 but v3 became a development set. False-positive rate on real case evidence could not be measured (the synthetic data has no memos). The semantic detector is opt-in per context (`SEMANTIC_TRIPWIRE=true` in the API; off in research scripts so earlier results are unchanged).

### 4.3 Static checks
`make lint` (ruff, 100-column, per-file exemptions for tests, experiments, regex and template files), `ruff format --check`, `make typecheck` (mypy strict, pydantic plugin), frontend `tsc --noEmit`, CI workflow running all of them plus the frontend build.

---

## 5. Every training run and its result

All training used synthetic data, on the laptop unless noted, single CPU thread for numeric libraries (ADR 0002).

### 5.1 KYC name reranker (`HistGradientBoostingClassifier`)
Fit on the train split (50% of entities; dev 20%, test 30%, entity-disjoint) of the 4,000-customer benchmark, seeds 42 and 43. Whole benchmark 118 s. Effect: F1 0.560 → 0.714 (seed 42) and 0.580 → 0.735 (seed 43) over the hybrid; no gain once structured verification is added (EXP-KYC-01, finding 4).

### 5.2 Anomaly models (medium preset, 250,000 rows; fit time / epochs)

| model | seed 42 | seed 43 |
|---|---|---|
| logistic regression | 0.2 s | 0.4 s |
| random forest | 8.8 s | 8.8 s |
| XGBoost | 2.2 s | 2.7 s |
| XGBoost, basic features | 1.0 s | 1.2 s |
| isolation forest | 0.4 s | 0.4 s |
| autoencoder | 7.5 s, 40 epochs | 6.8 s, 40 epochs |
| MLP | 5.6 s, 25 epochs (early stopping) | 6.1 s, 27 epochs |
| GRU sequence | 44.1 s, 8 epochs | 46.6 s, 8 epochs |
| GRU hybrid | 73.6 s, 13 epochs | 53.5 s, 9 epochs |

Whole benchmark about 185 s per seed. Thresholds chosen on validation only; scalers fit on train only. Results in 4.2.

### 5.3 Anomaly model inside the workflow
The XGBoost scorer is trained at startup on the labelled history before its cutoff (one-off setup about 14 s including the index and model load). Cases used for evaluation are restricted to after its training period (trained until 2025-04-25; held-out cases from 2025-05-29).

### 5.4 Semantic injection tripwire (logistic regression on MiniLM embeddings)
`scripts/train_tripwire.py`: 75 attack wordings (sets v1-v3) against 109 negatives (benign memos plus first sentences of knowledge-base sections); C = 10, decision threshold 0.7, class-balanced, fixed before v4 was written. Artifact `knowledge_base/ingestion/tripwire_model.json` (384 weights, bias, training-set hash). Result on the blind v4 set in 4.2 (recall 0.79, 0/20 false positives).

### 5.5 LoRA fine-tuning of Qwen2.5-0.5B: **not completed**
What exists: teacher (targets derived from evidence only, never from labels; never ESCALATE; placeholder confidence 0.6 REVIEW / 0.7 CLEAR), dataset (train 242 incl. 54 injected, val 22, eval 24, eval-injected 8; customer-disjoint; time split; train and eval injection wordings disjoint), trainer (LoRA r=16, alpha 32, dropout 0.05 on q,k,v,o,gate,up,down; loss on assistant tokens only; grad-accumulation 8; lr 2e-4 cosine with warmup; 2 epochs; gradient checkpointing; bf16, or float32 on a CUDA GPU without bf16).

What was attempted on the laptop and why it stopped:
1. A LoRA probe on the 1.5B model thrashed memory; the user chose the 0.5B instead.
2. A 2-step probe of the 0.5B produced no adapter and no timing output (it stalled after loading weights).
3. A foreground rerun was still running after 10 minutes at about 3% CPU with roughly 64 MB of free memory; it was killed.
4. A diagnostic measured sequence lengths (min 1,806, median 1,970, max 2,342 tokens) and a single forward pass with loss on the GPU raised an out-of-memory error at 9.06 GiB.

**No LoRA adapter has been trained and there are no fine-tuning results.** The fine-tuned arms of RQ6 are reported as PENDING everywhere (`results-summary.md`).

### 5.6 Inference runs (not training, listed for completeness)
Qwen2.5-1.5B fp16 on Metal: EXP-LLM-01 (28 cases, 19.3 s/case), EXP-AGENTS-01/02 (6 cases), EXP-LLM-02 reference arms (24 cases × 2 + 8 injected, about 32 s/case). Qwen2.5-0.5B: base arms (24 × 2 + 8 injected, about 10 s/case, 624 s total) and the RQ5 run (24 cases × 2 variants, 15.8 s / 24.9 s).

---

## 6. Corrections and negative results kept on the record

- RQ1: the BM25 + dense hybrid did not beat its parts.
- RQ2: no reliable winner among the top models; the temporal models' geo-change advantage is probably a confound.
- RQ3: RAG did not measurably improve grounding (already at ceiling); it lowered the 0.5B's format validity (0.83 → 0.67, inconclusive).
- RQ4: the engine floor helps; the unsupported-claim rule added nothing on this data.
- RQ5: inconclusive at n = 24; latency direction flipped between experiments.
- RQ6: pending.
- Injection: prompt-level defences failed (3 of 4 flips); the floor is the real defence; the original regex tripwire was weak (8/21) and its tuned gains were mostly overfitting; the semantic detector reaches 0.79 on a blind set and is still only a tripwire.
- Residual risk: about a third of behavioural problems reach no engine flag, so for those cases the advisory decision rests on the language model alone.
- Retracted/corrected statements are listed in 3.4.

---

## 7. What is left

Everything is built. Three things remain and all need something the laptop cannot provide:

1. **Train the LoRA adapter and run the fine-tuned arms (RQ6).** Use the notebook [notebooks/train_lora_colab.ipynb](../notebooks/train_lora_colab.ipynb) (free Colab T4, about 30-60 minutes; Kaggle works too) or a GPU PC ([architecture/finetune-on-gpu-pc.md](architecture/finetune-on-gpu-pc.md)). The notebook has not been run on a GPU by the author; its install line was verified in a clean environment. Then:
   - download `finguard_results.zip`; copy the adapter folder to `llm/fine_tuning/adapters/qwen0.5b-lora-v1/` and `finetune_eval.json` to `evaluation/reports/llm/finetune_eval.json`;
   - run `make results-report` (RQ6 fills in by itself), and ask for the EXP-FT-01 results write-up (its protocol and pre-stated hypotheses are already in experiments.md).
2. **Build and run the Docker images once** (`API_SECRET_KEY=$(openssl rand -base64 48) docker compose up --build`). The daemon was not running here, so only the compose file's syntax was validated; report any failure and it will be fixed.
3. **Optional:** enable GitHub Actions so CI runs on the repository; produce a larger, independently written wording set and question set (small n and one-author bias are the main weaknesses of the evaluations).

Known limitations that will remain: all data synthetic; small evaluation sets (24 cases); one seed for the LLM experiments; tripwires are heuristics; QLoRA is impossible on macOS (plain LoRA is used).

---

## 8. How to run the project

### 8.1 Setup
Requirements: Python 3.11, Node 20+ (22 used), optionally Docker.
```bash
git clone https://github.com/Ashmit-A-Rawat/FINGUARD-MajorProject.git FINGUARD && cd FINGUARD
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api,data,ml,kyc,rag,llm]"      # `make install` installs only the dev extra
cp .env.example .env                                  # edit API_SECRET_KEY etc.
export OMP_NUM_THREADS=1                              # macOS: required (the Makefile also sets PYTHONPATH=.)
```

### 8.2 Data and checks
```bash
make data            # small synthetic dataset, about 2 s (data/synthetic/small, git-ignored)
make pipeline        # validated consolidation
make test            # 407 backend tests
make lint typecheck  # ruff + mypy
make frontend-test   # 33 frontend tests
```

### 8.3 Reproduce the results
```bash
make kyc-benchmark            # ~2-5 min   (also: --seed 43 via the script)
make anomaly-benchmark        # ~3 min     (generates the medium preset)
make reconciliation-eval      # ~1 min
make rag-benchmark            # ~1 min
make ablation                 # seconds    EXP-ABL-01
make adversarial              # seconds    EXP-ADV-01
PYTHONPATH=. python experiments/adversarial/run_tripwire_eval.py --label final    # EXP-ADV-02
make guardrail-eval           # needs the EXP-LLM-01 report
make results-report           # regenerates docs/research/results-summary.md
```
LLM experiments need model weights (8.4). Base-model arms on a laptop: `python experiments/llm/run_finetune_eval.py --out evaluation/reports/llm/finetune_eval_base_laptop.json` (about 10 min for 0.5B); reference arms: add `--reference-only --reference-model llm/models/qwen2.5-1.5b-instruct`; RQ5: `python experiments/agents/run_orchestration_ablation.py`. Full table with times: [research/reproduce.md](research/reproduce.md).

### 8.4 Language models (never downloaded implicitly)
- `make hardware` estimates what fits your machine.
- Put weights under `llm/models/<name>/` (git-ignored): `qwen2.5-0.5b-instruct` and/or `qwen2.5-1.5b-instruct` (folders with `model.safetensors`, `config.json`, `tokenizer.json`, ...). Download with `huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir llm/models/qwen2.5-0.5b-instruct`, or `scripts/fetch_weights.sh <url> <file>` on a flaky connection.
- In `.env`: `LLM_PROVIDER=mock` (default: no model, output labelled MOCK) or `qwen`; `LLM_MODEL=llm/models/qwen2.5-1.5b-instruct`; optional `LLM_ADAPTER=llm/fine_tuning/adapters/qwen0.5b-lora-v1` once trained; `LLM_DEVICE=auto|cpu|mps|cuda`.
- Checks: `make llm-check` (structured-output check), `make agents-demo` (multi-agent demo).

### 8.5 API and web UI (local)
```bash
FINGUARD_PASSWORD='a long passphrase (12+ chars)' python scripts/create_user.py alice analyst   # roles: analyst | auditor | admin
make api                 # http://localhost:8000  (docs at /docs, health at /health; the engine loads in the background)
make frontend-dev        # http://localhost:5173  (proxies /api to the API)
```
Log in as the user you created, open a case for a customer/transaction, read the ten panels, then sign off. The UI shows an advisory banner above AI output and labels mock output. Roles: analyst decides and sees personal details (each view is audited); auditor and admin cannot decide cases. API reference: [api/api.md](api/api.md).

### 8.6 Containers
```bash
export API_SECRET_KEY=$(openssl rand -base64 48)
docker compose up --build                        # web UI on http://localhost:8080
docker compose exec -e FINGUARD_PASSWORD='choose-one-12+' api python scripts/create_user.py alice analyst
```
Model weights are mounted read-only from `./llm/models`; default `LLM_PROVIDER=mock`. Details and production notes (TLS, database roles): [architecture/deployment-architecture.md](architecture/deployment-architecture.md). **Not yet built by the author (see 7).**

### 8.7 Training the adapter online
Open `notebooks/train_lora_colab.ipynb` in Google Colab (Runtime > GPU) and run the cells top to bottom (steps 1-8), then follow section 7 item 1.

### 8.8 Troubleshooting
- `No module named 'backend'` → use `make`, or `PYTHONPATH=. python ...`, or `chflags -R nohidden .venv`.
- Exit code 139 or a hang at 0% CPU → `OMP_NUM_THREADS=1` (ADR 0002).
- API refuses to start outside development → set `API_SECRET_KEY` to 32+ random characters.
- Model not found → weights are never downloaded automatically; place them (8.4) or keep `LLM_PROVIDER=mock`.
- Semantic tripwire off → it needs the MiniLM embedder it was trained with; with any other embedder it is disabled automatically.

---

## 9. Repository map

| Path | Contents |
|---|---|
| `backend/app/` | FastAPI app: `api/`, `core/` (config, security, runtime), `database/`, `schemas/`, `services/` (audit chain, case store, engine, jobs) |
| `data_pipeline/` | synthetic generator, validation, consolidation |
| `kyc/`, `anomaly_detection/`, `reconciliation/` | the three deterministic/ML engines |
| `knowledge_base/` | documents, ingestion (loader, sanitiser, semantic tripwire), chunking, retrieval, vector store |
| `llm/` | providers, prompts, RAG glue, schemas, hardware, `fine_tuning/` (teacher, dataset, trainer) |
| `agents/`, `guardrails/` | workflow, agents, audit; evidence validator, policy checks, self-critique |
| `evaluation/`, `experiments/` | metrics, ablations, golden and adversarial datasets, reports; one runner per experiment |
| `frontend/` | React UI and tests, Dockerfile |
| `scripts/` | data generation, pipeline, user creation, hardware, weights, dataset build, training, tripwire training, results report |
| `data/finetune/` | committed fine-tuning dataset (synthetic) |
| `deploy/`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` | deployment and CI |
| `notebooks/` | Colab training notebook |
| `docs/` | this file, PROGRESS.md, architecture/, decisions/ (3 ADRs), research/, api/ |
