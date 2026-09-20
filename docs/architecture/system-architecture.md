# System Architecture

Status: **Phase 1 skeleton.** Sections are filled in as each phase lands.

## Layers (build order = dependency order)

1. Data: synthetic banking data, schemas, ingestion, consolidation (Phases 2-3)
2. Engines: KYC entity resolution, anomaly detection, reconciliation (Phases 4-6)
3. Knowledge: banking knowledge base + RAG retrieval (Phase 7)
4. LLM: provider abstraction (mock / Qwen / Mistral), structured output (Phase 8)
5. Agents: deterministic state-machine orchestration (Phase 9)
6. Guardrails: evidence validation, self-critique (Phase 10)
7. Human review UI + audit trail (Phase 11)

## Principles

- Deterministic engines produce evidence; the LLM only explains and investigates it.
- Every AI claim must cite evidence IDs; unsupported claims route to human REVIEW.
- Retrieved documents are DATA, never instructions.
- No real financial actions (e.g. account freezing) are ever taken automatically.
- All data is synthetic and labelled as such.


## Reconciliation engine (Phase 6, `reconciliation/`)

Deterministic and rule-based, deliberately not ML: it answers "do these two records agree?", where every
finding must be explainable and reproducible.

`matcher.py` pairs a transaction with its ledger record(s) by transaction id (earliest posting is *primary*, the
rest are *extras*). `rules.py` holds ten rules with stable ids. `discrepancy_detector.py` applies them and
returns findings in rule-id order. `service.py` exposes `reconcile`, `reconcile_many` and `reconcile_case`
(a case's focus + context transactions, judged at the case's `as_of` time). Source records are never modified;
results are new objects, and the config is frozen.

| Rule | Field | Severity | Group |
|---|---|---|---|
| REC-001 | no ledger entry | high | core |
| REC-002 | more than one ledger entry (duplicate posting) | high | core |
| REC-003 | amount differs by more than 1 cent | low / medium / high by relative size (1% / 10%) | core |
| REC-004 | currency differs | high | core |
| REC-005 | reference id missing on exactly one side | medium | core |
| REC-006 | reference ids both present but different | high | core |
| REC-007 | posted before the transaction | high | core |
| REC-008 | posted more than 3 days (T+3) after | medium; high beyond 3x the limit | core |
| REC-009 | ledger settlement failed | medium | status |
| REC-010 | still pending more than 3 days later (needs `as_of`) | medium | status |

Every finding carries `field`, `expected`, `actual`, `difference`, `severity`, `rule_id`, `explanation`.
Amounts are compared in integer cents so float noise cannot create findings. All tolerances live in
`ReconciliationConfig` (amount tolerance and severity cut-offs, max posting lag, status checks).

**Not compared: sender and receiver.** The ledger schema carries no counterparty fields, so those two comparisons
in the original spec are impossible without a schema change. Adding them is a Phase-2 schema decision.

**Status rules are a separate group.** Settlement `failed` and stale `pending` are legitimate reconciliation
signals, but the synthetic generator assigns statuses randomly (95% settled / 4% pending / 1% failed), so these
rules fire on about 5% of ordinary rows. They can be switched off with `check_settlement_status=False`, and
evaluation reports them separately from the core rules.


## Multi-agent workflow (Phase 9, `agents/`)

Orchestration is a deterministic state machine ([ADR 0003](../decisions/0003-explicit-state-machine-not-langgraph.md)); **no LLM decides what
happens next.** Only one agent calls a language model, and the model is advisory and untrusted (see EXP-LLM-01).

```
CASE_CREATED -> DATA_READY -> KYC_ANALYZED -> ANOMALY_ANALYZED -> RECONCILED -> EVIDENCE_RETRIEVED
   -> INVESTIGATION_GENERATED -> SELF_CRITIQUED -> HUMAN_REVIEW -> CLOSED   (CLOSED only by a named human)
```

| Agent | Responsibility (and nothing else) | Input -> output (`agents/contracts.py`) | LLM |
|---|---|---|---|
| Coordinator | order, state, audit, failure routing, human sign-off rules; builds the canonical case | (drives all) | no |
| Auditor | KYC verification (matcher: BM25 + dense + DOB/address) and anomaly scoring (XGBoost, top score drivers) | `KYCAnalysisInput -> KYCAnalysisOutput`; `AnomalyAnalysisInput -> AnomalyAnalysisOutput` | no |
| Reconciliation | transaction vs ledger rules for the case | `ReconciliationInput -> ReconciliationOutput` | no |
| Investigator | targeted knowledge retrieval, then a structured advisory investigation | `RetrievalInput -> RetrievalOutput`; `InvestigationInput -> InvestigationResult` | **yes** |
| Reviewer / critic | verifies every claim against the cited evidence and applies the deterministic policy floor (`guardrails/`, see [LLM architecture](llm-architecture.md#guardrails-phase-10-guardrails)); plug-in `Validator`s | `ReviewInput -> ReviewOutput` | no (deliberately) |
| Report | assembles the explainable report | `ReportInput -> CaseReport` | no |

Rules that keep responsibilities from overlapping: an agent reads only its declared input and returns only its declared output; **only the
coordinator merges outputs into `CaseState`**; evidence ids are unique per case (a clash raises); agents never touch each other's engines.

**Fail-safe behaviour.** A step exception marks `failed_step`, produces a best-effort report and moves the case to HUMAN_REVIEW; an invalid LLM answer
(after the bounded repair loop) leaves `investigation=None` and the report says so. Nothing continues silently past a failure and nothing is auto-closed.

**Human sign-off (`CaseWorkflow.sign_off`).** Requires a named reviewer and a written reason; ESCALATE requires a second, *different* reviewer (four-eyes);
only valid from HUMAN_REVIEW; every sign-off is an audit event.

**Audit trail.** One `AuditEvent` per step: id, timestamp, case id, request id, actor, action, from/to status, duration, ok/error and small structured details
(provider, mock flag, counts), which contain no names or document numbers (tested). Each event is also emitted as a JSON log line.

**Report honesty.** Engine facts (deterministic) and model statements are separate lists with an `origin`. The report is marked UNVALIDATED until the
Phase 10 validators exist, MOCK when a mock provider produced it, flags anomaly scores from inside the model's training period, lists reference text withheld as suspicious,
and carries a **consistency warning** when the model proposes CLEAR while engines report a high-severity finding (enforcement is Phase 10).

**Single-agent baseline** (`agents/baseline.py`): one function, one LLM call, one generic retrieval query, no state machine, audit or reviewer; same engines and provider.
It exists for the RQ5 ablation in Phase 13.

**Limits.** `build_context` trains the anomaly model at startup from labelled history (fine for the small dataset; a real deployment would load a persisted model). Cases must be
from the held-out period to avoid optimistic scores (flagged when not). The KYC step matches the case's documents against the customer master with the un-reranked hybrid+structured
system (no fitted reranker needed).


## Human review: API, UI and audit trail (Phase 11)

```
React UI (Vite, TS, Tailwind) --/api--> FastAPI --> SQLAlchemy (SQLite dev / PostgreSQL) 
                                           |--> EngineService (loads the analysis engine in a background thread)
                                           '--> single worker thread runs CaseWorkflow per case
```

**Persistence.** `cases` holds the workflow state as JSON plus denormalised inbox columns; `audit_events` holds the trail; `users` holds credentials.
The analysis engine (dataset, KYC index, anomaly model, knowledge base, LLM) loads in the background at startup; the API is usable immediately and reports `engine_ready`.
Sign-off does not need the engine, so decisions can be recorded even while it is loading.

**Audit trail is append-only and tamper-evident.**
- Every workflow step, every human action *and every read* (viewing a case, viewing personal details, opening the trail) is an event with actor, action, status transition, duration and small details (no names or document numbers).
- Sequence numbers are assigned **by the database**, not by the caller, so the background job, an analyst viewing the case and a sign-off can interleave without dropping or duplicating events.
- Events are hash-chained per case (`hash = SHA-256(previous hash || canonical event)`). Editing, deleting or reordering a stored event breaks the chain, and the API reports the first bad event; the UI shows **CHAIN BROKEN**.
  The ORM also refuses updates and deletes of audit rows. Both are defence in depth; someone with full write access to the database could recompute the chain, so production must
  give the API's database role INSERT and SELECT only on `audit_events` (see deployment architecture).

**Security model.** scrypt password hashes; signed tokens with the role re-read from the database each request; login throttling; role checks on every endpoint;
strict request validation; security headers (`nosniff`, `X-Frame-Options: DENY`, `no-store` on API responses); CORS restricted to configured origins and to GET/POST; a strong `API_SECRET_KEY` is
mandatory outside development (the app refuses to start otherwise). Personal details are returned only to analysts.

**The UI** shows the ten panels from the brief: case inbox, customer/KYC, transaction timeline, anomaly analysis, reconciliation, retrieved evidence, AI investigation, self-critique and decision and audit trail.
It never lets model text look authoritative: an advisory banner sits above the AI investigation, every claim carries its verdict and reason, mock output is labelled MOCK, the model's confidence is labelled uncalibrated, and choosing CLEAR against the advisory requires an
explicit acknowledgement. The server re-checks every rule the UI mirrors.

**Known limits.** No user-management UI or MFA; tokens live in `sessionStorage` (readable by any script that runs in the page, so a strict Content-Security-Policy belongs at the reverse proxy); a token stays valid until it expires unless the user is deactivated;
the login throttle is per process; TLS must be terminated in front of the API (dev runs over HTTP); the job queue is in-process (a restart loses queued jobs; the case then shows `queued`); PostgreSQL support is written but was not exercised here (no Docker daemon
was available), so only SQLite is tested.
