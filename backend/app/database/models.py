"""ORM tables. The audit table is append-only (enforced in code, with hash chaining on top)."""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.base import Base


class UserRow(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(16))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class CaseRow(Base):
    __tablename__ = "cases"
    case_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(64), index=True)
    focus_transactions: Mapped[str] = mapped_column(Text)  # JSON list of ids
    status: Mapped[str] = mapped_column(String(32), index=True)
    job_state: Mapped[str] = mapped_column(String(16), index=True)  # queued|running|done|failed
    job_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    advisory_decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    proposed_decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    validated: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    warnings: Mapped[int] = mapped_column(Integer, default=0)
    is_mock: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    state_json: Mapped[str] = mapped_column(Text)
    timeline_json: Mapped[str] = mapped_column(Text, default="[]")
    customer_json: Mapped[str] = mapped_column(Text, default="{}")
    pending_escalation_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pending_escalation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class AuditRow(Base):
    __tablename__ = "audit_events"
    __table_args__ = (UniqueConstraint("case_id", "seq", name="uq_audit_case_seq"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.case_id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    request_id: Mapped[str] = mapped_column(String(64))
    actor: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(64))
    from_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    to_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    duration_ms: Mapped[float] = mapped_column(Float)
    ok: Mapped[bool] = mapped_column(Boolean)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    details_json: Mapped[str] = mapped_column(Text)
    prev_hash: Mapped[str] = mapped_column(String(64))
    hash: Mapped[str] = mapped_column(String(64))


class AuditImmutableError(RuntimeError):
    """An attempt to modify or delete an audit event."""


@event.listens_for(AuditRow, "before_update")
def _block_audit_update(*_: object) -> None:
    raise AuditImmutableError("audit events are append-only and cannot be modified")


@event.listens_for(AuditRow, "before_delete")
def _block_audit_delete(*_: object) -> None:
    raise AuditImmutableError("audit events are append-only and cannot be deleted")
