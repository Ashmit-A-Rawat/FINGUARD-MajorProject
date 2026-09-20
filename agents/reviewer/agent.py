"""Reviewer / critic agent. Responsibility: check the investigation BEFORE a human sees it.

With a ``SelfCritique`` it verifies every claim against the cited evidence and applies the
deterministic policy floor (see ``guardrails``). Without one (the "no self-critique" ablation) it
says so plainly: ``validated=False``. Either way it never approves anything on its own.
"""

from collections.abc import Sequence
from typing import Protocol

from agents.contracts import CheckResult, ReviewInput, ReviewOutput
from guardrails.self_critique import SelfCritique


class Validator(Protocol):
    check_id: str

    def validate(self, inp: ReviewInput) -> list[CheckResult]: ...


class ReviewerAgent:
    name = "reviewer"

    def __init__(
        self, validators: Sequence[Validator] = (), critique: SelfCritique | None = None
    ) -> None:
        self._validators = list(validators)
        self._critique = critique

    def review(self, inp: ReviewInput) -> ReviewOutput:
        report = (
            self._critique.review(inp.evidence, inp.investigation, inp.knowledge_chunks)
            if self._critique
            else None
        )
        validator_checks = [c for v in self._validators for c in v.validate(inp)]
        if report is None and not validator_checks:
            return ReviewOutput(
                validated=False,
                notes=["No automated validators are configured; the investigation is UNVALIDATED."],
            )
        policy_checks = (
            [
                CheckResult(check_id=f.code, passed=not f.is_violation, detail=f.detail)
                for f in report.policy.flags
            ]
            if report
            else []
        )
        notes = list(report.notes) if report else []
        if report is None and inp.investigation is None:
            notes.append("No investigation was produced to validate.")
        base_ok = report.validated if report else inp.investigation is not None
        validated = base_ok and all(c.passed for c in validator_checks)
        return ReviewOutput(
            validated=validated,
            checks=validator_checks + policy_checks,
            notes=notes,
            critique=report,
        )
