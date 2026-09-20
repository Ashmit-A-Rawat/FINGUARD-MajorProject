"""Reviewer / critic agent. Responsibility: check the investigation BEFORE a human sees it.

Phase 9 delivers the interface and wiring; the actual validators (evidence support, policy floor,
self-critique) are Phase 10. With none configured the agent says so plainly: ``validated=False``.
It never approves anything by default.
"""

from collections.abc import Sequence
from typing import Protocol

from agents.contracts import CheckResult, ReviewInput, ReviewOutput


class Validator(Protocol):
    check_id: str

    def validate(self, inp: ReviewInput) -> list[CheckResult]: ...


class ReviewerAgent:
    name = "reviewer"

    def __init__(self, validators: Sequence[Validator] = ()) -> None:
        self._validators = list(validators)

    def review(self, inp: ReviewInput) -> ReviewOutput:
        if not self._validators:
            return ReviewOutput(
                validated=False,
                notes=["No automated validators are configured; the investigation is UNVALIDATED."],
            )
        checks = [c for v in self._validators for c in v.validate(inp)]
        notes = [] if inp.investigation else ["No investigation was produced to validate."]
        validated = inp.investigation is not None and all(c.passed for c in checks)
        return ReviewOutput(validated=validated, checks=checks, notes=notes)
