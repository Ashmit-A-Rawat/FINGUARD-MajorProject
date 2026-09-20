"""Persistence of cases: state, denormalised inbox columns and the audit trail."""

import json
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agents.audit import AuditEvent
from agents.state import CaseState
from backend.app.database.models import AuditRow, CaseRow
from backend.app.services.audit_chain import append_events, row_to_event

STATE_EXCLUDE = {"canonical_case", "audit_trail"}


def _dump_state(state: CaseState) -> str:
    return state.model_dump_json(exclude=STATE_EXCLUDE)


def _denormalise(row: CaseRow, state: CaseState, now: datetime) -> None:
    report = state.report
    row.status = state.status.value
    row.updated_at = now
    row.advisory_decision = (
        report.advisory_decision.value if report and report.advisory_decision else None
    )
    row.proposed_decision = (
        report.proposed_decision.value if report and report.proposed_decision else None
    )
    row.validated = report.validated if report else None
    row.warnings = len(report.warnings) if report else 0
    row.is_mock = state.investigation_meta.is_mock if state.investigation_meta else None
    row.state_json = _dump_state(state)


def create_case_row(session: Session, state: CaseState, created_by: str, now: datetime) -> CaseRow:
    row = CaseRow(
        case_id=state.case_id,
        customer_id=state.customer_id,
        focus_transactions=json.dumps(state.focus_transaction_ids),
        status=state.status.value,
        job_state="queued",
        created_by=created_by,
        created_at=now,
        updated_at=now,
        state_json=_dump_state(state),
    )
    session.add(row)
    session.flush()
    append_events(session, state.case_id, list(state.audit_trail))  # brand-new case: all events new
    return row


def save_state(
    session: Session,
    state: CaseState,
    now: datetime,
    *,
    job_state: str | None = None,
    job_error: str | None = None,
    timeline: list[dict[str, Any]] | None = None,
    customer: dict[str, Any] | None = None,
    new_events: list[AuditEvent] | None = None,
) -> CaseRow:
    """Persist the state; ``new_events`` are the audit events produced since the last save."""
    row = session.get(CaseRow, state.case_id)
    if row is None:
        raise KeyError(state.case_id)
    _denormalise(row, state, now)
    if job_state is not None:
        row.job_state, row.job_error = job_state, job_error
    if timeline is not None:
        row.timeline_json = json.dumps(timeline, default=str)
    if customer is not None:
        row.customer_json = json.dumps(customer, default=str)
    append_events(session, state.case_id, new_events or [])
    return row


def add_audit_event(
    session: Session, case_id: str, request_id: str, event: AuditEvent
) -> AuditEvent:
    """Append a single event (for example an access log) to a case's trail."""
    return append_events(session, case_id, [event.model_copy(update={"request_id": request_id})])[0]


def load_state(session: Session, case_id: str) -> tuple[CaseRow, CaseState] | None:
    row = session.get(CaseRow, case_id)
    if row is None:
        return None
    state = CaseState.model_validate_json(row.state_json)
    state.audit_trail = load_audit(session, case_id)
    return row, state


def load_audit(session: Session, case_id: str) -> list[AuditEvent]:
    rows = (
        session.execute(select(AuditRow).where(AuditRow.case_id == case_id).order_by(AuditRow.seq))
        .scalars()
        .all()
    )
    return [row_to_event(r) for r in rows]


def list_cases(
    session: Session, status: str | None, limit: int, offset: int
) -> tuple[list[CaseRow], int]:
    query = select(CaseRow)
    count = select(func.count()).select_from(CaseRow)
    if status:
        query, count = query.where(CaseRow.status == status), count.where(CaseRow.status == status)
    rows = (
        session.execute(query.order_by(CaseRow.created_at.desc()).limit(limit).offset(offset))
        .scalars()
        .all()
    )
    return list(rows), int(session.execute(count).scalar_one())
