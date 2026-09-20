"""The case state that flows through the workflow. Only the coordinator mutates it."""

from datetime import datetime

from pydantic import BaseModel, Field

from agents.audit import AuditEvent
from agents.contracts import (
    AnomalyFinding,
    CaseReport,
    InvestigationMeta,
    KYCDocumentFinding,
    ReviewOutput,
)
from backend.app.schemas.domain import CaseStatus, Decision, Evidence, ReconciliationResult
from data_pipeline.consolidation.models import CanonicalCase
from knowledge_base.models import RetrievedChunk
from llm.schemas import InvestigationOutput


class SignOff(BaseModel):
    reviewer: str
    second_reviewer: str | None
    decision: Decision
    reason: str
    at: datetime


class CaseState(BaseModel):
    case_id: str
    request_id: str
    customer_id: str
    focus_transaction_ids: list[str]
    context_days: int = 30
    status: CaseStatus = CaseStatus.CASE_CREATED
    created_at: datetime
    canonical_case: CanonicalCase | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    kyc_findings: list[KYCDocumentFinding] = Field(default_factory=list)
    anomaly_findings: list[AnomalyFinding] = Field(default_factory=list)
    reconciliation_results: list[ReconciliationResult] = Field(default_factory=list)
    retrieval_queries: list[str] = Field(default_factory=list)
    knowledge_chunks: list[RetrievedChunk] = Field(default_factory=list)
    investigation: InvestigationOutput | None = None
    investigation_meta: InvestigationMeta | None = None
    review: ReviewOutput | None = None
    report: CaseReport | None = None
    failed_step: str | None = None  # set when a step raised; the case then goes to human review
    sign_off: SignOff | None = None
    audit_trail: list[AuditEvent] = Field(default_factory=list)

    def add_evidence(self, items: list[Evidence]) -> None:
        known = {e.evidence_id for e in self.evidence}
        clash = known & {e.evidence_id for e in items}
        if clash:
            raise ValueError(f"duplicate evidence ids: {sorted(clash)}")
        self.evidence.extend(items)
