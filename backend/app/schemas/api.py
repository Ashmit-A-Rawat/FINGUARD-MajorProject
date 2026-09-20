"""API models. Inputs are validated tightly; identity is never taken from a request body."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agents.contracts import (
    AnomalyFinding,
    CaseReport,
    InvestigationMeta,
    KYCDocumentFinding,
    ReviewOutput,
)
from agents.state import SignOff
from backend.app.core.security import Role
from backend.app.schemas.domain import Decision, Evidence, ReconciliationResult
from knowledge_base.models import RetrievedChunk
from llm.schemas import InvestigationOutput

ID_PATTERN = r"^[A-Za-z0-9_.:\-]{1,100}$"
USERNAME_PATTERN = r"^[a-z0-9._-]{3,64}$"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginRequest(Strict):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: Role
    expires_in_minutes: int


class UserOut(BaseModel):
    username: str
    role: Role


class CreateUserRequest(Strict):
    username: str = Field(pattern=USERNAME_PATTERN)
    password: str = Field(min_length=12, max_length=256)
    role: Role


class CaseSummary(BaseModel):
    case_id: str
    customer_id: str
    focus_transaction_ids: list[str]
    status: str
    job_state: str
    job_error: str | None
    advisory_decision: Decision | None
    proposed_decision: Decision | None
    validated: bool | None
    warnings: int
    is_mock: bool | None
    created_by: str
    created_at: datetime
    updated_at: datetime
    pending_escalation_by: str | None


class CaseListResponse(BaseModel):
    items: list[CaseSummary]
    total: int


class CreateCaseRequest(Strict):
    customer_id: str = Field(pattern=ID_PATTERN)
    transaction_ids: list[str] | None = Field(default=None, max_length=20)
    context_days: int = Field(default=30, ge=1, le=365)


class SignOffRequest(Strict):
    decision: Decision
    reason: str = Field(min_length=1, max_length=2000)
    # Required to choose CLEAR against a REVIEW/ESCALATE advisory (an explicit, audited override).
    acknowledge_advisory_override: bool = False


class EscalationDecisionRequest(Strict):
    reason: str = Field(min_length=1, max_length=2000)


class PendingEscalation(BaseModel):
    proposed_by: str
    reason: str


class CaseDetail(BaseModel):
    summary: CaseSummary
    customer: dict[str, Any]  # personal details are removed unless the caller is an analyst
    timeline: list[dict[str, Any]]
    anomaly_findings: list[AnomalyFinding]
    kyc_findings: list[KYCDocumentFinding]
    reconciliation_results: list[ReconciliationResult]
    evidence: list[Evidence]
    knowledge_chunks: list[RetrievedChunk]
    retrieval_queries: list[str]
    investigation: InvestigationOutput | None
    investigation_meta: InvestigationMeta | None
    review: ReviewOutput | None
    report: CaseReport | None
    sign_off: SignOff | None
    pending_escalation: PendingEscalation | None


class AuditEventOut(BaseModel):
    event_id: int
    timestamp: datetime
    actor: str
    action: str
    from_status: str | None
    to_status: str | None
    duration_ms: float
    ok: bool
    error: str | None
    details: dict[str, Any]
    prev_hash: str
    hash: str


class ChainStatus(BaseModel):
    ok: bool
    events: int
    first_bad_seq: int | None
    reason: str


class AuditResponse(BaseModel):
    events: list[AuditEventOut]
    chain: ChainStatus


class CandidateOut(BaseModel):
    customer_id: str
    transaction_id: str
    timestamp: datetime
    amount: float
    currency: str
    has_case: bool
