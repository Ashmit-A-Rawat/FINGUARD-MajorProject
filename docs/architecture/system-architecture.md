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
| Reviewer / critic | checks the investigation before a human sees it; plug-in `Validator`s | `ReviewInput -> ReviewOutput` | (Phase 10) |
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
