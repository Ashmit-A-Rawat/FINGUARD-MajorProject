"""Human sign-off rules, independent of the (heavy) analysis engine.

A case is CLOSED only here, by a named human: reviewer and reason are mandatory; ESCALATE needs a
second, DIFFERENT reviewer (four-eyes). Callers (the API) must pass identities taken from
authentication, never from free text supplied by a client.
"""

from agents.audit import Clock, record_event
from agents.state import CaseState, SignOff
from backend.app.schemas.domain import CaseStatus, Decision


class SignOffError(ValueError):
    """A sign-off violated the human-review rules."""


def sign_off_case(
    state: CaseState,
    clock: Clock,
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
    now = clock()
    state.sign_off = SignOff(
        reviewer=reviewer.strip(),
        second_reviewer=second_reviewer,
        decision=decision,
        reason=reason.strip(),
        at=now,
    )
    state.status = CaseStatus.CLOSED
    record_event(
        state,
        clock,
        actor=f"human:{reviewer.strip()}",
        action="sign_off",
        from_status=CaseStatus.HUMAN_REVIEW,
        to_status=CaseStatus.CLOSED,
        details={"decision": decision.value},
    )
    return state
