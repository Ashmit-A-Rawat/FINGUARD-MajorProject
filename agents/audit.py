"""Audit trail: an append-only list of events for a case."""

from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from backend.app.schemas.domain import CaseStatus

Clock = Callable[[], datetime]


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
