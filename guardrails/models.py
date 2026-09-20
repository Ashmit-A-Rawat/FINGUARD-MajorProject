"""Result types of the guardrails (claim checks, policy flags)."""

from typing import Literal

from pydantic import BaseModel, Field

from backend.app.schemas.domain import Decision

SupportStatus = Literal["supported", "unsupported", "unverifiable"]


class ClaimCheck(BaseModel):
    """Verdict for one claim: the claim, what it cites, and why it is (not) supported."""

    index: int  # position among the investigation's findings; -1 for the summary
    claim: str
    kind: str  # fact | inference | summary
    evidence_ids: list[str]
    status: SupportStatus
    reason: str

    @property
    def supported(self) -> bool:
        return self.status == "supported"


class PolicyFlag(BaseModel):
    code: str
    detail: str
    is_violation: bool  # a violation means the model output cannot be accepted as is


class PolicyResult(BaseModel):
    flags: list[PolicyFlag] = Field(default_factory=list)
    floor: Decision | None = None  # minimum decision implied by deterministic engine findings
    floor_reasons: list[str] = Field(default_factory=list)


class CritiqueReport(BaseModel):
    """Everything the self-critique found. Deterministic: produced without asking an LLM."""

    claim_checks: list[ClaimCheck] = Field(default_factory=list)
    summary_grounded: bool = True
    summary_reason: str = ""
    policy: PolicyResult = Field(default_factory=PolicyResult)
    model_decision: Decision | None = None  # what the model proposed
    guardrail_decision: Decision | None = None  # never lower than the engine floor
    adjustments: list[str] = Field(default_factory=list)
    validated: bool = False
    notes: list[str] = Field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {
            s: sum(c.status == s for c in self.claim_checks)
            for s in ("supported", "unsupported", "unverifiable")
        }
