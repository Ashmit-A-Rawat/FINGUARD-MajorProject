"""Background processing of a case by the analysis engine."""

import logging

from agents.audit import Clock
from agents.state import CaseState
from backend.app.database.base import SessionFactory
from backend.app.database.models import CaseRow
from backend.app.services.case_store import save_state
from backend.app.services.case_views import build_customer, build_timeline
from backend.app.services.engine import EngineService

logger = logging.getLogger(__name__)


def run_case_job(
    engine: EngineService,
    session_factory: SessionFactory,
    state: CaseState,
    persisted_events: int,
    clock: Clock,
) -> None:
    """Run the workflow and persist the result. ``persisted_events`` is how many audit events of
    ``state`` are already stored; only later ones are appended."""
    assert engine.workflow is not None
    try:
        with session_factory() as session:
            save_state(session, state, clock(), job_state="running")
            session.commit()
        state = engine.workflow.run(state)
        with session_factory() as session:
            save_state(
                session,
                state,
                clock(),
                job_state="done",
                timeline=build_timeline(state),
                customer=build_customer(state),
                new_events=state.audit_trail[persisted_events:],
            )
            session.commit()
    except Exception as exc:  # noqa: BLE001 - a crashed job must be visible, not silent
        logger.exception("case job failed: %s", state.case_id)
        with session_factory() as session:
            row = session.get(CaseRow, state.case_id)
            if row is not None:
                row.job_state, row.job_error = "failed", f"{type(exc).__name__}: {exc}"[:500]
                session.commit()
