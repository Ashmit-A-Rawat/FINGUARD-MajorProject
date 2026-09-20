"""Audit trail: an append-only list of events for a case."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, Field

from backend.app.schemas.domain import CaseStatus

Clock = Callable[[], datetime]


class CaseStateLike(Protocol):
    case_id: str
    request_id: str
    audit_trail: list["AuditEvent"]


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)  # naive UTC, like every other timestamp here


class AuditEvent(BaseModel):
    event_id: int
    timestamp: datetime
    case_id: str
    request_id: str
    actor: str  # agent name, or "human:<reviewer>"
    action: str
    from_status: CaseStatus | None
    to_status: CaseStatus | None
    duration_ms: float
    ok: bool
    error: str | None = None
    details: dict[str, object] = Field(default_factory=dict)  # small, no personal data


def record_event(
    state: "CaseStateLike",
    clock: Clock,
    *,
    actor: str,
    action: str,
    from_status: CaseStatus | None,
    to_status: CaseStatus | None,
    duration_ms: float = 0.0,
    ok: bool = True,
    error: str | None = None,
    details: dict[str, object] | None = None,
) -> AuditEvent:
    """Append one event to a case's audit trail (ids are sequential per case) and return it."""
    event = AuditEvent(
        event_id=len(state.audit_trail) + 1,
        timestamp=clock(),
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
    return event
