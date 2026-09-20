"""Single-agent baseline for the RQ5 ablation (Phase 13): one monolithic function, one LLM call.

No state machine, no per-agent handoffs, no audit events, no reviewer or report stage, and a single
generic retrieval query. Same engines, same evidence builders, same provider, so any difference in
accuracy or latency comes from the orchestration and targeted retrieval, not from different tools.
"""

import time

from pydantic import BaseModel

from agents.context import AgentContext
from agents.contracts import InvestigationMeta
from backend.app.schemas.domain import Evidence
from llm.prompts.investigation import build_investigation_prompt
from llm.rag.evidence import (
    anomaly_model_evidence,
    behaviour_evidence,
    customer_evidence,
    reconciliation_evidence,
    transaction_evidence,
)
from llm.rag.structured import generate_structured
from llm.schemas import InvestigationOutput


class BaselineResult(BaseModel):
    case_id: str
    evidence_ids: list[str]
    investigation: InvestigationOutput | None
    meta: InvestigationMeta
    wall_seconds: float


class SingleAgentBaseline:
    name = "single_agent"

    def __init__(self, ctx: AgentContext) -> None:
        self._ctx = ctx

    def run(
        self, customer_id: str, transaction_ids: list[str] | None = None, context_days: int = 30
    ) -> BaselineResult:
        started = time.perf_counter()
        ctx, now = self._ctx, self._ctx.clock()
        history = ctx.store.transactions_by_customer[customer_id]
        focus = transaction_ids or [history[-1].transaction_id]
        case = ctx.store.build_case(customer_id, focus, context_days=context_days)

        evidence: list[Evidence] = [customer_evidence(case.customer.customer, now)]
        for canonical in case.focus_transactions:
            tx = canonical.transaction
            evidence += [transaction_evidence(tx, now), behaviour_evidence(tx, history, now)]
            evidence.append(anomaly_model_evidence(ctx.anomaly.score(tx.transaction_id), now))
        for result in ctx.reconciliation.reconcile_case(case):
            if result.transaction_id in focus:
                evidence += reconciliation_evidence(result, now)

        query = " ".join(e.description for e in evidence)[:400]  # one generic query
        chunks = ctx.knowledge_base.retriever.search(query, k=2)
        bundle = build_investigation_prompt(evidence, chunks)
        result_ = generate_structured(ctx.llm, bundle.messages, InvestigationOutput)
        first = result_.attempts[0].result
        return BaselineResult(
            case_id=case.case_id,
            evidence_ids=bundle.evidence_ids,
            investigation=result_.value,
            meta=InvestigationMeta(
                prompt_version=bundle.prompt_version,
                nonce=bundle.nonce,
                provider=first.provider,
                model=first.model,
                is_mock=result_.is_mock,
                attempts=len(result_.attempts),
                errors=[a.error for a in result_.attempts if a.error],
                latency_s=result_.total_latency_s,
                knowledge_chunk_ids=bundle.knowledge_chunk_ids,
                excluded_suspicious_chunks=bundle.excluded_suspicious,
            ),
            wall_seconds=time.perf_counter() - started,
        )
