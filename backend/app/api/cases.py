"""Case endpoints. Every identity comes from the authenticated token; every access is audited."""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from agents.audit import AuditEvent, record_event, utc_now
from agents.coordinator.signoff import SignOffError, sign_off_case
from agents.state import CaseState
from backend.app.api.deps import CurrentUser, get_session, require_roles
from backend.app.core.security import Role
from backend.app.database.models import AuditRow, CaseRow
from backend.app.schemas.api import (
    ID_PATTERN,
    AuditEventOut,
    AuditResponse,
    CandidateOut,
    CaseDetail,
    CaseListResponse,
    CaseSummary,
    ChainStatus,
    CreateCaseRequest,
    EscalationDecisionRequest,
    PendingEscalation,
    SignOffRequest,
)
from backend.app.schemas.domain import CaseStatus, Decision
from backend.app.services.audit_chain import verify_chain
from backend.app.services.case_store import (
    add_audit_event,
    create_case_row,
    list_cases,
    load_state,
    save_state,
)
from backend.app.services.jobs import run_case_job

router = APIRouter(prefix="/api", tags=["cases"])
CaseId = Annotated[str, Path(pattern=ID_PATTERN)]
Reader = Annotated[CurrentUser, Depends(require_roles(Role.ANALYST, Role.AUDITOR, Role.ADMIN))]
Analyst = Annotated[CurrentUser, Depends(require_roles(Role.ANALYST))]
SessionDep = Annotated[Session, Depends(get_session)]


def _summary(row: CaseRow) -> CaseSummary:
    return CaseSummary(
        case_id=row.case_id,
        customer_id=row.customer_id,
        focus_transaction_ids=json.loads(row.focus_transactions),
        status=row.status,
        job_state=row.job_state,
        job_error=row.job_error,
        advisory_decision=Decision(row.advisory_decision) if row.advisory_decision else None,
        proposed_decision=Decision(row.proposed_decision) if row.proposed_decision else None,
        validated=row.validated,
        warnings=row.warnings,
        is_mock=row.is_mock,
        created_by=row.created_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
        pending_escalation_by=row.pending_escalation_by,
    )


def _detail(row: CaseRow, state: CaseState, role: Role) -> CaseDetail:
    customer = json.loads(row.customer_json)
    if role != Role.ANALYST:  # personal details only for the reviewers who decide cases
        customer.pop("details", None)
    return CaseDetail(
        summary=_summary(row),
        customer=customer,
        timeline=json.loads(row.timeline_json),
        anomaly_findings=state.anomaly_findings,
        kyc_findings=state.kyc_findings,
        reconciliation_results=state.reconciliation_results,
        evidence=state.evidence,
        knowledge_chunks=state.knowledge_chunks,
        retrieval_queries=state.retrieval_queries,
        investigation=state.investigation,
        investigation_meta=state.investigation_meta,
        review=state.review,
        report=state.report,
        sign_off=state.sign_off,
        pending_escalation=(
            PendingEscalation(
                proposed_by=row.pending_escalation_by, reason=row.pending_escalation_reason or ""
            )
            if row.pending_escalation_by
            else None
        ),
    )


def _load(session: Session, case_id: str) -> tuple[CaseRow, CaseState]:
    loaded = load_state(session, case_id)
    if loaded is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "case not found")
    return loaded


def _human_event(user: CurrentUser, action: str, state: CaseState, **details: object) -> AuditEvent:
    """A human action recorded on the trail (event_id is assigned by the database)."""
    return AuditEvent(
        event_id=0,
        timestamp=utc_now(),
        case_id=state.case_id,
        request_id=state.request_id,
        actor=f"human:{user.username}",
        action=action,
        from_status=state.status,
        to_status=state.status,
        duration_ms=0.0,
        ok=True,
        details=dict(details),
    )


@router.get("/cases", response_model=CaseListResponse)
def list_all(
    _: Reader,
    session: SessionDep,
    status_filter: Annotated[CaseStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> CaseListResponse:
    rows, total = list_cases(session, status_filter.value if status_filter else None, limit, offset)
    return CaseListResponse(items=[_summary(r) for r in rows], total=total)


@router.post("/cases", response_model=CaseSummary, status_code=status.HTTP_202_ACCEPTED)
def create_case(
    body: CreateCaseRequest, request: Request, user: Analyst, session: SessionDep
) -> CaseSummary:
    engine = request.app.state.engine
    if engine is None or not engine.ready:
        detail = (
            engine.error if engine is not None and engine.error else "analysis engine is loading"
        )
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail)
    try:
        state = engine.workflow.create_case(
            body.customer_id, body.transaction_ids, body.context_days
        )
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown customer") from None
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from None
    if session.get(CaseRow, state.case_id) is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "a case for this customer and transaction exists"
        )
    record_event(
        state,
        utc_now,
        actor=f"human:{user.username}",
        action="request_case",
        from_status=None,
        to_status=state.status,
    )
    row = create_case_row(session, state, user.username, utc_now())
    persisted = len(state.audit_trail)
    session.commit()  # the worker must be able to see the row
    engine.submit(
        lambda: run_case_job(engine, request.app.state.session_factory, state, persisted, utc_now)
    )
    return _summary(row)


@router.get("/cases/{case_id}", response_model=CaseDetail)
def get_case(case_id: CaseId, user: Reader, session: SessionDep) -> CaseDetail:
    row, state = _load(session, case_id)
    add_audit_event(session, case_id, state.request_id, _human_event(user, "view_case", state))
    if user.role == Role.ANALYST and row.customer_json not in ("{}", ""):
        add_audit_event(
            session, case_id, state.request_id, _human_event(user, "view_customer_details", state)
        )
    return _detail(row, state, user.role)


@router.post("/cases/{case_id}/sign-off", response_model=CaseDetail)
def sign_off(
    case_id: CaseId, body: SignOffRequest, user: Analyst, session: SessionDep
) -> CaseDetail:
    row, state = _load(session, case_id)
    if state.status != CaseStatus.HUMAN_REVIEW:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"case is {state.status.value}, not awaiting review"
        )
    if row.pending_escalation_by:
        raise HTTPException(status.HTTP_409_CONFLICT, "an escalation is awaiting a second reviewer")
    n0 = len(state.audit_trail)
    advisory = state.report.advisory_decision if state.report else None
    if body.decision == Decision.CLEAR and advisory in (Decision.REVIEW, Decision.ESCALATE):
        if not body.acknowledge_advisory_override:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"the advisory decision is {advisory.value}; choosing CLEAR requires "
                "acknowledge_advisory_override=true (the override is recorded)",
            )
        state.audit_trail.append(
            _human_event(
                user,
                "advisory_override_acknowledged",
                state,
                advisory=advisory.value,
                chosen=Decision.CLEAR.value,
            )
        )
    if body.decision == Decision.ESCALATE:
        row.pending_escalation_by, row.pending_escalation_reason = user.username, body.reason
        state.audit_trail.append(_human_event(user, "escalation_proposed", state))
    else:
        try:
            sign_off_case(state, utc_now, user.username, body.decision, body.reason)
        except SignOffError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None
    row = save_state(session, state, utc_now(), new_events=state.audit_trail[n0:])
    return _detail(row, state, user.role)


def _pending(row: CaseRow, user: CurrentUser) -> str:
    if not row.pending_escalation_by:
        raise HTTPException(status.HTTP_409_CONFLICT, "no escalation is awaiting a second reviewer")
    if row.pending_escalation_by.casefold() == user.username.casefold():
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "the second reviewer must be a different person (four-eyes rule)",
        )
    return row.pending_escalation_by


@router.post("/cases/{case_id}/sign-off/confirm", response_model=CaseDetail)
def confirm_escalation(
    case_id: CaseId, body: EscalationDecisionRequest, user: Analyst, session: SessionDep
) -> CaseDetail:
    row, state = _load(session, case_id)
    proposer = _pending(row, user)
    n0 = len(state.audit_trail)
    state.audit_trail.append(
        _human_event(user, "escalation_confirmed", state, confirm_reason=body.reason[:200])
    )
    try:
        sign_off_case(
            state,
            utc_now,
            proposer,
            Decision.ESCALATE,
            row.pending_escalation_reason or "",
            second_reviewer=user.username,
        )
    except SignOffError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None
    row.pending_escalation_by = row.pending_escalation_reason = None
    row = save_state(session, state, utc_now(), new_events=state.audit_trail[n0:])
    return _detail(row, state, user.role)


@router.post("/cases/{case_id}/sign-off/reject", response_model=CaseDetail)
def reject_escalation(
    case_id: CaseId, body: EscalationDecisionRequest, user: Analyst, session: SessionDep
) -> CaseDetail:
    """A second reviewer who disagrees sends the case back to review (the case stays open)."""
    row, state = _load(session, case_id)
    _pending(row, user)
    n0 = len(state.audit_trail)
    state.audit_trail.append(
        _human_event(user, "escalation_rejected", state, reason=body.reason[:200])
    )
    row.pending_escalation_by = row.pending_escalation_reason = None
    row = save_state(session, state, utc_now(), new_events=state.audit_trail[n0:])
    return _detail(row, state, user.role)


@router.get("/cases/{case_id}/audit", response_model=AuditResponse)
def audit_trail(case_id: CaseId, user: Reader, session: SessionDep) -> AuditResponse:
    row, state = _load(session, case_id)
    add_audit_event(
        session, case_id, state.request_id, _human_event(user, "view_audit_trail", state)
    )
    session.flush()
    rows = (
        session.execute(select(AuditRow).where(AuditRow.case_id == case_id).order_by(AuditRow.seq))
        .scalars()
        .all()
    )
    report = verify_chain(session, case_id)
    events = [
        AuditEventOut(
            event_id=r.seq,
            timestamp=r.timestamp,
            actor=r.actor,
            action=r.action,
            from_status=r.from_status,
            to_status=r.to_status,
            duration_ms=r.duration_ms,
            ok=r.ok,
            error=r.error,
            details=json.loads(r.details_json),
            prev_hash=r.prev_hash,
            hash=r.hash,
        )
        for r in rows
    ]
    return AuditResponse(
        events=events,
        chain=ChainStatus(
            ok=report.ok,
            events=report.events,
            first_bad_seq=report.first_bad_seq,
            reason=report.reason,
        ),
    )


@router.get("/candidates", response_model=list[CandidateOut])
def candidates(
    request: Request,
    _: Analyst,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
) -> list[CandidateOut]:
    """Recent transactions from the held-out period that could be opened as cases."""
    engine = request.app.state.engine
    if engine is None or not engine.ready or engine.context is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "analysis engine is loading")
    ctx = engine.context
    txs = [
        t
        for v in ctx.store.transactions_by_customer.values()
        for t in v
        if t.timestamp >= ctx.anomaly.trained_until.to_pydatetime()
    ]
    txs.sort(key=lambda t: t.timestamp, reverse=True)
    existing = set(session.execute(select(CaseRow.case_id)).scalars())
    return [
        CandidateOut(
            customer_id=t.customer_id,
            transaction_id=t.transaction_id,
            timestamp=t.timestamp,
            amount=t.amount,
            currency=t.currency,
            has_case=f"CASE-{t.customer_id}-{t.transaction_id}" in existing,
        )
        for t in txs[:limit]
    ]
