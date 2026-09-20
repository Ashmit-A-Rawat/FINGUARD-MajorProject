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
