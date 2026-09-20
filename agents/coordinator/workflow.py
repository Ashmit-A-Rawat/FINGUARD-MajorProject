"""Deterministic case workflow: an explicit state machine, not an LLM-driven planner.

    CASE_CREATED -> DATA_READY -> KYC_ANALYZED -> ANOMALY_ANALYZED -> RECONCILED
      -> EVIDENCE_RETRIEVED -> INVESTIGATION_GENERATED -> SELF_CRITIQUED -> HUMAN_REVIEW -> CLOSED

The order is fixed by ``STEPS``. Every step is timed and written to the audit trail. A failing
step never aborts silently: the case moves to HUMAN_REVIEW with ``failed_step`` set and a
best-effort report. A case reaches CLOSED ONLY through ``sign_off`` by a named human.
"""

import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from agents.audit import AuditEvent
from agents.auditor.agent import AuditorAgent
from agents.context import AgentContext
from agents.contracts import (
    AnomalyAnalysisInput,
    InvestigationInput,
    KYCAnalysisInput,
    ReconciliationInput,
    ReportInput,
    RetrievalInput,
    ReviewInput,
)
from agents.investigator.agent import InvestigationAgent
from agents.reconciliation.agent import ReconciliationAgent
from agents.report_generator.agent import ReportAgent
from agents.reviewer.agent import ReviewerAgent
from agents.state import CaseState, SignOff
from backend.app.schemas.domain import CaseStatus, Decision
from data_pipeline.consolidation.models import CanonicalCase
from guardrails.self_critique import SelfCritique

logger = logging.getLogger("fin_guard.workflow")
Details = dict[str, object]


class SignOffError(ValueError):
    """A sign-off violated the human-review rules."""


@dataclass(frozen=True)
class Step:
    name: str
    actor: str
    from_status: CaseStatus
    to_status: CaseStatus
    run: Callable[[CaseState, datetime], Details]


class CaseWorkflow:
    def __init__(self, ctx: AgentContext, reviewer: ReviewerAgent | None = None) -> None:
        self._ctx = ctx
        self._auditor = AuditorAgent(ctx)
        self._reconciliation = ReconciliationAgent(ctx)
        self._investigator = InvestigationAgent(ctx)
        self._reviewer = reviewer or ReviewerAgent(
            critique=SelfCritique()
        )  # guardrails on by default
        self._reporter = ReportAgent()
        S = CaseStatus
        self.steps: tuple[Step, ...] = (
            Step("prepare_data", "coordinator", S.CASE_CREATED, S.DATA_READY, self._prepare),
            Step("analyze_kyc", "auditor", S.DATA_READY, S.KYC_ANALYZED, self._kyc),
            Step("analyze_anomalies", "auditor", S.KYC_ANALYZED, S.ANOMALY_ANALYZED, self._anomaly),
            Step("reconcile", "reconciliation", S.ANOMALY_ANALYZED, S.RECONCILED, self._reconcile),
            Step(
                "retrieve_knowledge",
                "investigator",
                S.RECONCILED,
                S.EVIDENCE_RETRIEVED,
                self._retrieve,
            ),
            Step(
                "investigate",
                "investigator",
                S.EVIDENCE_RETRIEVED,
                S.INVESTIGATION_GENERATED,
                self._investigate,
            ),
            Step("review", "reviewer", S.INVESTIGATION_GENERATED, S.SELF_CRITIQUED, self._review),
            Step(
                "generate_report",
                "report_generator",
                S.SELF_CRITIQUED,
                S.HUMAN_REVIEW,
                self._report,
            ),
        )

    # ---------------------------------------------------------------- public API
    def create_case(
        self,
        customer_id: str,
        transaction_ids: list[str] | None = None,
        context_days: int = 30,
        request_id: str | None = None,
    ) -> CaseState:
        store = self._ctx.store
        if customer_id not in store.customers:
            raise KeyError(f"unknown customer {customer_id}")
        history = store.transactions_by_customer.get(customer_id, [])
        focus = transaction_ids or ([history[-1].transaction_id] if history else [])
        if not focus:
            raise ValueError(f"customer {customer_id} has no transactions")
        known = {t.transaction_id for t in history}
        if unknown := [t for t in focus if t not in known]:
            raise ValueError(f"transactions not found for customer {customer_id}: {unknown}")
        state = CaseState(
            case_id=f"CASE-{customer_id}-{focus[0]}",
            request_id=request_id or uuid.uuid4().hex[:12],
            customer_id=customer_id,
            focus_transaction_ids=focus,
            context_days=context_days,
            created_at=self._ctx.clock(),
        )
        self._log(state, "coordinator", "create_case", None, CaseStatus.CASE_CREATED, 0.0, True)
        return state

    def run(self, state: CaseState) -> CaseState:
        """Advance until HUMAN_REVIEW (never beyond)."""
        for step in self.steps:
            if state.status != step.from_status:
                raise RuntimeError(
                    f"{step.name} expects {step.from_status}, case is {state.status}"
                )
            started = time.perf_counter()
            try:
                details = step.run(state, self._ctx.clock())
            except Exception as exc:  # noqa: BLE001 - any step failure must fail safe to a human
                self._fail(state, step, started, exc)
                return state
            state.status = step.to_status
            self._log(
                state,
                step.actor,
                step.name,
                step.from_status,
                step.to_status,
                1000 * (time.perf_counter() - started),
                True,
                details=details,
            )
        return state

    def sign_off(
        self,
        state: CaseState,
        reviewer: str,
        decision: Decision,
        reason: str,
        second_reviewer: str | None = None,
    ) -> CaseState:
        if state.status != CaseStatus.HUMAN_REVIEW:
            raise SignOffError(f"case is {state.status}, not awaiting human review")
        if not reviewer.strip() or not reason.strip():
            raise SignOffError("a named reviewer and a written reason are required")
        if decision == Decision.ESCALATE:
            second = (second_reviewer or "").strip()
            if not second or second.casefold() == reviewer.strip().casefold():
                raise SignOffError("ESCALATE needs a second, different reviewer (four-eyes rule)")
        now = self._ctx.clock()
        state.sign_off = SignOff(
            reviewer=reviewer.strip(),
            second_reviewer=second_reviewer,
            decision=decision,
            reason=reason.strip(),
            at=now,
        )
        state.status = CaseStatus.CLOSED
        self._log(
            state,
            f"human:{reviewer.strip()}",
            "sign_off",
            CaseStatus.HUMAN_REVIEW,
            CaseStatus.CLOSED,
            0.0,
            True,
            details={"decision": decision.value},
        )
        return state

    # ---------------------------------------------------------------- steps
    def _prepare(self, state: CaseState, now: datetime) -> Details:
        state.canonical_case = self._ctx.store.build_case(
            state.customer_id,
            state.focus_transaction_ids,
            context_days=state.context_days,
            case_id=state.case_id,
        )
        return {
            "focus": len(state.focus_transaction_ids),
            "context": len(state.canonical_case.context_transactions),
        }

    def _case(self, state: CaseState) -> CanonicalCase:
        if state.canonical_case is None:
            raise RuntimeError("case data not prepared")
        return state.canonical_case

    def _kyc(self, state: CaseState, now: datetime) -> Details:
        out = self._auditor.analyze_kyc(KYCAnalysisInput(case=self._case(state)), now)
        state.kyc_findings = out.findings
        state.add_evidence(out.evidence)
        return {"documents": len(out.findings)}

    def _anomaly(self, state: CaseState, now: datetime) -> Details:
        out = self._auditor.analyze_anomalies(AnomalyAnalysisInput(case=self._case(state)), now)
        state.anomaly_findings = out.findings
        state.add_evidence(out.evidence)
        return {"flagged": sum(f.flagged for f in out.findings)}

    def _reconcile(self, state: CaseState, now: datetime) -> Details:
        out = self._reconciliation.reconcile(ReconciliationInput(case=self._case(state)), now)
        state.reconciliation_results = out.results
        state.add_evidence(out.evidence)
        return {
            "transactions": len(out.results),
            "with_discrepancy": sum(r.status == "discrepancy" for r in out.results),
        }

    def _retrieve(self, state: CaseState, now: datetime) -> Details:
        out = self._investigator.retrieve(RetrievalInput(evidence=state.evidence))
        state.retrieval_queries, state.knowledge_chunks = out.queries, out.chunks
        return {"queries": len(out.queries), "chunks": [c.chunk_id for c in out.chunks]}

    def _investigate(self, state: CaseState, now: datetime) -> Details:
        out = self._investigator.investigate(
            InvestigationInput(evidence=state.evidence, chunks=state.knowledge_chunks)
        )
        state.investigation, state.investigation_meta = out.investigation, out.meta
        return {
            "valid": out.investigation is not None,
            "attempts": out.meta.attempts,
            "provider": out.meta.provider,
            "is_mock": out.meta.is_mock,
            "llm_latency_s": round(out.meta.latency_s, 3),
        }

    def _review(self, state: CaseState, now: datetime) -> Details:
        out = self._reviewer.review(
            ReviewInput(
                evidence=state.evidence,
                investigation=state.investigation,
                knowledge_chunks=state.knowledge_chunks,
            )
        )
        state.review = out
        details: Details = {"validated": out.validated, "checks": len(out.checks)}
        if out.critique is not None:
            details.update(out.critique.counts)
            details["guardrail_decision"] = (
                out.critique.guardrail_decision.value if out.critique.guardrail_decision else None
            )
        return details

    def _report(self, state: CaseState, now: datetime) -> Details:
        state.report = self._reporter.generate(self._report_input(state, now))
        return {"warnings": len(state.report.warnings)}

    def _report_input(self, state: CaseState, now: datetime) -> ReportInput:
        return ReportInput(
            case_id=state.case_id,
            customer_id=state.customer_id,
            focus_transaction_ids=state.focus_transaction_ids,
            generated_at=now,
            evidence=state.evidence,
            kyc_findings=state.kyc_findings,
            anomaly_findings=state.anomaly_findings,
            reconciliation_results=state.reconciliation_results,
            knowledge_chunks=state.knowledge_chunks,
            investigation=state.investigation,
            investigation_meta=state.investigation_meta,
            review=state.review,
            failed_step=state.failed_step,
        )

    # ---------------------------------------------------------------- bookkeeping
    def _fail(self, state: CaseState, step: Step, started: float, exc: Exception) -> None:
        state.failed_step = step.name
        self._log(
            state,
            step.actor,
            step.name,
            step.from_status,
            step.from_status,
            1000 * (time.perf_counter() - started),
            False,
            error=f"{type(exc).__name__}: {exc}",
        )
        previous = state.status
        state.status = CaseStatus.HUMAN_REVIEW  # fail safe: a person looks at it
        try:  # best-effort report from whatever exists
            state.report = self._reporter.generate(self._report_input(state, self._ctx.clock()))
        except Exception as report_exc:  # noqa: BLE001
            self._log(
                state,
                "report_generator",
                "generate_report",
                previous,
                state.status,
                0.0,
                False,
                error=f"{type(report_exc).__name__}: {report_exc}",
            )
        self._log(
            state,
            "coordinator",
            "route_to_human_review",
            previous,
            CaseStatus.HUMAN_REVIEW,
            0.0,
            True,
            details={"failed_step": step.name},
        )

    def _log(
        self,
        state: CaseState,
        actor: str,
        action: str,
        from_status: CaseStatus | None,
        to_status: CaseStatus | None,
        duration_ms: float,
        ok: bool,
        error: str | None = None,
        details: Details | None = None,
    ) -> None:
        event = AuditEvent(
            event_id=len(state.audit_trail) + 1,
            timestamp=self._ctx.clock(),
            case_id=state.case_id,
            request_id=state.request_id,
            actor=actor,
            action=action,
            from_status=from_status,
            to_status=to_status,
            duration_ms=duration_ms,
            ok=ok,
            error=error,
            details=details or {},
        )
        state.audit_trail.append(event)
        logger.info(json.dumps(event.model_dump(mode="json"), default=str))
