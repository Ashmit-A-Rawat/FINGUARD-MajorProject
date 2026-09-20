"""Reconciliation agent. Responsibility: reconcile the case's transactions with the ledger.
Deterministic wrapper around the rule engine; never modifies source records."""

from collections import Counter
from datetime import datetime

from agents.context import AgentContext
from agents.contracts import ReconciliationInput, ReconciliationOutput
from backend.app.schemas.domain import Evidence, EvidenceSource
from llm.rag.evidence import reconciliation_evidence


class ReconciliationAgent:
    name = "reconciliation"

    def __init__(self, ctx: AgentContext) -> None:
        self._ctx = ctx

    def reconcile(self, inp: ReconciliationInput, now: datetime) -> ReconciliationOutput:
        case = inp.case
        results = self._ctx.reconciliation.reconcile_case(case)
        by_id = {r.transaction_id: r for r in results}
        focus_ids = [c.transaction.transaction_id for c in case.focus_transactions]
        evidence: list[Evidence] = []
        for transaction_id in focus_ids:
            evidence += reconciliation_evidence(by_id[transaction_id], now)
        context = [r for r in results if r.transaction_id not in set(focus_ids)]
        rules = Counter(d.rule_id for r in context for d in r.discrepancies)
        evidence.append(
            Evidence(
                evidence_id=f"REC-CONTEXT-{case.case_id}",
                source=EvidenceSource.RECONCILIATION,
                description=(
                    f"{sum(bool(r.discrepancies) for r in context)} of {len(context)} recent "
                    f"transactions of this customer have reconciliation findings"
                ),
                payload={"context_transactions": len(context), "findings_by_rule": dict(rules)},
                created_at=now,
            )
        )
        return ReconciliationOutput(results=results, evidence=evidence)
