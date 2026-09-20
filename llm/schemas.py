"""Structured investigation output the LLM must produce (validated with Pydantic)."""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from backend.app.schemas.domain import Decision


def normalize_citation(reference: str) -> str:
    """Canonical id of a cited item: strip brackets and the ``E:`` / ``K:`` token prefix.

    The prompt shows evidence as ``[E:TXN-1]`` and models (correctly) often cite ``E:TXN-1``; both
    forms mean evidence id ``TXN-1``. Validators must compare normalised ids.
    """
    ref = reference.strip().strip("[]").strip()
    return ref[2:] if ref[:2] in ("E:", "K:", "e:", "k:") else ref


class FindingKind(StrEnum):
    FACT = "fact"  # taken directly from the evidence
    INFERENCE = "inference"  # a conclusion drawn from facts


class Finding(BaseModel):
    statement: str = Field(min_length=1)
    kind: FindingKind
    evidence_ids: list[str] = Field(default_factory=list)  # ids the statement rests on

    @field_validator("kind", mode="before")
    @classmethod
    def _lower(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value


class InvestigationOutput(BaseModel):
    """Advisory only: a human reviewer makes the decision. Whether cited evidence really supports
    each finding is NOT checked here; that is the Phase 10 evidence validator's job."""

    summary: str = Field(min_length=1)
    findings: list[Finding]
    evidence: list[str]  # every evidence id the report relies on
    uncertainties: list[str]
    recommended_action: Decision
    recommendations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)  # the model's own estimate, NOT calibrated

    @field_validator("recommended_action", mode="before")
    @classmethod
    def _upper(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value
