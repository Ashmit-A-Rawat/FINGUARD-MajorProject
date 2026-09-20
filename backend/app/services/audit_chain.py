"""Tamper-evident audit trail: each event stores the hash of the previous one (per case).

hash_n = SHA-256(hash_{n-1} || canonical JSON of event n). Editing, deleting or reordering any
stored event breaks every later hash, which ``verify_chain`` detects. It shows that a trail was
altered after the fact; it cannot stop someone with full database write access from recomputing the
whole chain, so the database account used by the API must not have UPDATE/DELETE rights on this
table in production (see docs/architecture/deployment-architecture.md).
"""

import hashlib
import json
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from agents.audit import AuditEvent
from backend.app.database.models import AuditRow

GENESIS = "0" * 64


def _canonical(event: AuditEvent) -> bytes:
    payload = event.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


def _row_canonical(row: AuditRow) -> bytes:
    event = row_to_event(row)
    return _canonical(event)


def event_hash(prev_hash: str, event: AuditEvent) -> str:
    return hashlib.sha256(prev_hash.encode() + _canonical(event)).hexdigest()


def row_to_event(row: AuditRow) -> AuditEvent:
    return AuditEvent(
        event_id=row.seq,
        timestamp=row.timestamp,
        case_id=row.case_id,
        request_id=row.request_id,
        actor=row.actor,
        action=row.action,
        from_status=row.from_status,
        to_status=row.to_status,
        duration_ms=row.duration_ms,
        ok=row.ok,
        error=row.error,
        details=json.loads(row.details_json),
    )


def append_events(session: Session, case_id: str, events: list[AuditEvent]) -> list[AuditEvent]:
    """Append NEW events, assigning the next sequence numbers in the database.

    Sequence numbers come from the stored trail, never from the caller, so several writers (the
    background job, the API logging a view, a sign-off) can interleave without dropping or
    duplicating events: order is commit order. Returns the events with their final ``event_id``.
    """
    last = session.execute(
        select(AuditRow).where(AuditRow.case_id == case_id).order_by(AuditRow.seq.desc()).limit(1)
    ).scalar_one_or_none()
    seq, prev = (last.seq, last.hash) if last else (0, GENESIS)
    stored: list[AuditEvent] = []
    for event in events:
        seq += 1
        final = event.model_copy(update={"event_id": seq})
        digest = event_hash(prev, final)
        session.add(
            AuditRow(
                case_id=case_id,
                seq=seq,
                timestamp=final.timestamp,
                request_id=final.request_id,
                actor=final.actor,
                action=final.action,
                from_status=final.from_status.value if final.from_status else None,
                to_status=final.to_status.value if final.to_status else None,
                duration_ms=final.duration_ms,
                ok=final.ok,
                error=final.error,
                details_json=json.dumps(final.details, default=str),
                prev_hash=prev,
                hash=digest,
            )
        )
        prev = digest
        stored.append(final)
    session.flush()
    return stored


@dataclass
class ChainReport:
    ok: bool
    events: int
    first_bad_seq: int | None = None
    reason: str = ""


def verify_chain(session: Session, case_id: str) -> ChainReport:
    rows = (
        session.execute(select(AuditRow).where(AuditRow.case_id == case_id).order_by(AuditRow.seq))
        .scalars()
        .all()
    )
    prev = GENESIS
    for expected_seq, row in enumerate(rows, start=1):
        if row.seq != expected_seq:
            return ChainReport(False, len(rows), row.seq, "sequence gap or reordering")
        if row.prev_hash != prev:
            return ChainReport(False, len(rows), row.seq, "previous-hash link is broken")
        if hashlib.sha256(prev.encode() + _row_canonical(row)).hexdigest() != row.hash:
            return ChainReport(False, len(rows), row.seq, "event content does not match its hash")
        prev = row.hash
    return ChainReport(True, len(rows))
