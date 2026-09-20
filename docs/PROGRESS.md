# FIN-GUARD: progress log

Short record of what is built, what was measured, and what is still open. All data is synthetic.
Detail lives in `docs/research/experiments.md` (results) and `docs/architecture/` (design).
Updated at the end of every phase.

**Status: 11 of 14 phases done; phase 12 (fine-tuning) in progress.**

| # | Phase | State |
|---|-------|-------|
| 1 | Skeleton, config, health API | done |
| 2 | Synthetic data (customers, KYC, transactions, ledgers, injected faults) | done |
| 3 | Data pipeline + canonical case | done |
| 4 | KYC matching / entity resolution | done |
| 5 | Anomaly detection (XGBoost + rules, time-split) | done |
| 6 | Reconciliation rule engine | done |
| 7 | Knowledge base (ingest, sanitize, Chroma, retrieval, trust by directory) | done |
| 8 | LLM layer (mock / local HF provider, prompts, RAG, schemas, hardware check) | done |
| 9 | Multi-agent workflow (auditor, reconciliation, investigator, reviewer, reporter) | done |
| 10 | Guardrails (evidence validator, engine policy floor, self-critique) | done |
| 11 | API, React UI, human review, hash-chained audit trail | done |
| 12 | LoRA fine-tuning of Qwen2.5-0.5B | in progress |
| 13 | Evaluation and ablations (RQ3-RQ6, repeated runs, CIs) | todo |
| 14 | Deployment, Docker, final docs | todo |

## Design rules that hold everywhere
- Deterministic engines produce the evidence; the LLM is advisory and untrusted.
- Final decision = max(model, engine floor). The model can never lower risk, and is never forced to ESCALATE.
- Every model claim is checked against case evidence (supported / unsupported / unverifiable).
- Knowledge-base trust is set by source directory, not file content; untrusted chunks are excluded by default.
- Audit trail is hash-chained per case; sequence numbers come from the database.
- Escalation needs a second, authenticated reviewer (four-eyes).
- Mock outputs are always labelled mock. No result is reported that was not produced by the code.

## Phase 12 so far
- Teacher targets are built from evidence only, never from ground-truth labels. Never ESCALATE, fixed uncalibrated confidence.
- Dataset: train 242, val 22, eval 24, eval_injected 8; customer-disjoint; time-split (train model up to 2025-04-25, eval from 2025-05-29).
- Train and eval injection wordings are disjoint (eval includes one deliberately evasive wording).
- Trainer: LoRA r=16 on attention + MLP, loss on assistant tokens only, bf16 base, adapters git-ignored.
- `LocalHFProvider` can load an adapter and toggle it off, so base and tuned arms share one model.
- Four-arm eval script `experiments/llm/run_finetune_eval.py` is built and wired (checked with mock only). Dataset is committed.
- Training does not fit on the 8 GB laptop (~2,000-token sequences run MPS out of memory), so it will run on a GPU PC: see `docs/architecture/finetune-on-gpu-pc.md`.
- Still to do (on the PC): train, run the eval, bring back the report/adapter, write up EXP-FT-01.

## Corrections made along the way
- EXP-AGENTS-01 latency explanation retracted after it was found wrong.
- "2 of 4 injected pairs" sentence in EXP-GUARD-01 corrected.
- `.gitignore` rule was hiding `knowledge_base/vector_store/chroma_store.py`, so a fresh clone lacked it. Fixed and the file is now tracked.

## Known limitations
- QLoRA is not possible on macOS, so LoRA on a bf16 base is used instead.
- Small models (0.5B / 1.5B) are weak reasoners; the guardrails, not the model, provide safety.
- Injection tripwire is a heuristic; a rephrased attack can bypass it, and the engine floor is the real protection.
- Small evaluation sets (24 cases); results need bootstrap CIs and are not evidence of real-world performance.

## Housekeeping
Unused placeholders, empty packages, regenerable data and caches are deleted each phase. Removed so far:
`backend/app/models`, `data_pipeline/entity_resolution`, `scripts/download_model.py`, `.gitkeep` files,
empty `data/{raw,kyc,transactions,reconciliation}` and `notebooks/`, and generated `data/synthetic/medium*`.
