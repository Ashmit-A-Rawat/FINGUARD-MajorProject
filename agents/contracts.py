"""Typed inputs and outputs of every agent. An agent may read only its declared input and
writes only its declared output; the coordinator is the only component that merges outputs into
the case state (no hidden shared state between agents)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from backend.app.schemas.domain import Decision, Evidence, ReconciliationResult
from data_pipeline.consolidation.models import CanonicalCase
from knowledge_base.models import RetrievedChunk
from llm.schemas import InvestigationOutput


# ---- Auditor: KYC ---------------------------------------------------------------------------
class KYCCandidateSummary(BaseModel):
    candidate_id: str
    final_score: float
    is_match: bool
    contradictions: list[str] = Field(default_factory=list)


class KYCDocumentFinding(BaseModel):
    document_id: str
    own_record_rank: int | None  # rank of the customer's own record among candidates (1 = best)
    own_final_score: float | None
    own_confidence: str | None
    own_reasons: list[str] = Field(default_factory=list)
    own_contradictions: list[str] = Field(default_factory=list)
    other_strong_candidates: list[KYCCandidateSummary] = Field(default_factory=list)


class KYCAnalysisInput(BaseModel):
    case: CanonicalCase


class KYCAnalysisOutput(BaseModel):
    findings: list[KYCDocumentFinding]
    evidence: list[Evidence]


# ---- Auditor: anomaly -----------------------------------------------------------------------
class AnomalyFinding(BaseModel):
    transaction_id: str
    probability: float
    flagged: bool
    threshold: float
    top_drivers: list[tuple[str, float]]  # (feature, contribution to the score)
    in_training_period: bool  # True means the score is optimistic (model saw this period)
    model: str


class AnomalyAnalysisInput(BaseModel):
    case: CanonicalCase


class AnomalyAnalysisOutput(BaseModel):
    findings: list[AnomalyFinding]
    evidence: list[Evidence]


# ---- Reconciliation -------------------------------------------------------------------------
class ReconciliationInput(BaseModel):
    case: CanonicalCase


class ReconciliationOutput(BaseModel):
    results: list[ReconciliationResult]
    evidence: list[Evidence]


# ---- Investigator ---------------------------------------------------------------------------
class RetrievalInput(BaseModel):
    evidence: list[Evidence]


class RetrievalOutput(BaseModel):
    queries: list[str]
    chunks: list[RetrievedChunk]


class InvestigationInput(BaseModel):
    evidence: list[Evidence]
    chunks: list[RetrievedChunk]


class InvestigationMeta(BaseModel):
    prompt_version: str
    nonce: str
    provider: str
    model: str
    is_mock: bool
    attempts: int
    errors: list[str] = Field(default_factory=list)
    latency_s: float
    knowledge_chunk_ids: list[str] = Field(default_factory=list)
    excluded_suspicious_chunks: list[str] = Field(default_factory=list)


class InvestigationResult(BaseModel):
    investigation: InvestigationOutput | None  # None = the model failed to produce valid output
    meta: InvestigationMeta


# ---- Reviewer / critic ----------------------------------------------------------------------
class CheckResult(BaseModel):
    check_id: str
    passed: bool
    detail: str


class ReviewInput(BaseModel):
    evidence: list[Evidence]
    investigation: InvestigationOutput | None


class ReviewOutput(BaseModel):
    validated: bool  # False until real validators exist and have run
    checks: list[CheckResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---- Report ---------------------------------------------------------------------------------
class ReportItem(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    origin: Literal["engine", "model"]  # deterministic engine output vs model-generated text


class CaseReport(BaseModel):
    case_id: str
    customer_id: str
    focus_transaction_ids: list[str]
    generated_at: datetime
    advisory_only: Literal[True] = True
    is_synthetic: Literal[True] = True
    proposed_decision: Decision | None  # the MODEL's proposal; a human decides
    proposed_decision_note: str
    validated: bool
    engine_facts: list[ReportItem]  # deterministic engine outputs
    model_findings: list[ReportItem]  # unvalidated model statements
    recommendations: list[str]
    uncertainties: list[str]
    warnings: list[str]
    evidence_index: dict[str, str]  # evidence id -> description
    provenance: dict[str, object]


class ReportInput(BaseModel):
    case_id: str
    customer_id: str
    focus_transaction_ids: list[str]
    generated_at: datetime
    evidence: list[Evidence]
    kyc_findings: list[KYCDocumentFinding]
    anomaly_findings: list[AnomalyFinding]
    reconciliation_results: list[ReconciliationResult]
    knowledge_chunks: list[RetrievedChunk]
    investigation: InvestigationOutput | None
    investigation_meta: InvestigationMeta | None
    review: ReviewOutput | None
    failed_step: str | None
