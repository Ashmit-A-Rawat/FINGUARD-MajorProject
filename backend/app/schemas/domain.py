"""Core domain schemas shared by the data pipeline, engines and API.

Deliberately free of FastAPI imports so non-web code can depend on it.
Every data-bearing record carries ``is_synthetic=True``: this project has no real data.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CountryCode = Field(pattern=r"^[A-Z]{2}$")
CurrencyCode = Field(pattern=r"^[A-Z]{3}$")


class KYCStatus(StrEnum):
    VERIFIED = "verified"
    PENDING = "pending"
    EXPIRED = "expired"
    REJECTED = "rejected"


class AccountType(StrEnum):
    PERSONAL = "personal"
    BUSINESS = "business"
    SAVINGS = "savings"


class DocumentType(StrEnum):
    PASSPORT = "passport"
    NATIONAL_ID = "national_id"
    DRIVERS_LICENSE = "drivers_license"


class TransactionType(StrEnum):
    TRANSFER_OUT = "transfer_out"
    TRANSFER_IN = "transfer_in"
    PAYMENT = "payment"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"


class Channel(StrEnum):
    MOBILE = "mobile"
    WEB = "web"
    BRANCH = "branch"
    ATM = "atm"
    POS = "pos"


class SettlementStatus(StrEnum):
    SETTLED = "settled"
    PENDING = "pending"
    FAILED = "failed"


class CaseStatus(StrEnum):
    """Deterministic workflow states (see docs/architecture/system-architecture.md)."""

    CASE_CREATED = "case_created"
    DATA_READY = "data_ready"
    KYC_ANALYZED = "kyc_analyzed"
    ANOMALY_ANALYZED = "anomaly_analyzed"
    RECONCILED = "reconciled"
    EVIDENCE_RETRIEVED = "evidence_retrieved"
    INVESTIGATION_GENERATED = "investigation_generated"
    SELF_CRITIQUED = "self_critiqued"
    HUMAN_REVIEW = "human_review"
    CLOSED = "closed"


class Decision(StrEnum):
    CLEAR = "CLEAR"
    REVIEW = "REVIEW"
    ESCALATE = "ESCALATE"


class EvidenceSource(StrEnum):
    KYC = "kyc"
    ANOMALY = "anomaly"
    RECONCILIATION = "reconciliation"
    KNOWLEDGE_BASE = "knowledge_base"
    TRANSACTION = "transaction"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class _SyntheticRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_synthetic: Literal[True] = True


class Customer(_SyntheticRecord):
    customer_id: str
    name: str = Field(min_length=1)
    alternate_names: list[str] = Field(default_factory=list)
    date_of_birth: date
    address: str = Field(min_length=1)
    country: str = CountryCode
    occupation: str
    account_type: AccountType
    account_open_date: date
    account_age_days: int = Field(ge=0)
    kyc_status: KYCStatus

    @field_validator("alternate_names", mode="before")
    @classmethod
    def _split_alternate_names(cls, value: Any) -> Any:
        """Tabular files store alternate names as a '|'-joined string."""
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return value.split("|")
        return value


class KYCRecord(_SyntheticRecord):
    document_id: str
    customer_id: str
    name: str = Field(min_length=1)
    date_of_birth: date
    address: str = Field(min_length=1)
    document_type: DocumentType
    document_number: str = Field(min_length=1)
    issue_date: date
    expiry_date: date

    @field_validator("expiry_date")
    @classmethod
    def _expiry_after_issue(cls, value: date, info: Any) -> date:
        issue = info.data.get("issue_date")
        if issue is not None and value <= issue:
            raise ValueError("expiry_date must be after issue_date")
        return value


class Transaction(_SyntheticRecord):
    """Behavioural ground-truth labels are NOT stored here (see transaction_labels)."""

    transaction_id: str
    customer_id: str
    timestamp: datetime
    amount: float = Field(gt=0)
    currency: str = CurrencyCode
    transaction_type: TransactionType
    sender: str
    receiver: str
    sender_country: str = CountryCode
    receiver_country: str = CountryCode
    channel: Channel
    reference_id: str | None = None


class LedgerRecord(_SyntheticRecord):
    ledger_id: str
    transaction_id: str
    posted_amount: float
    currency: str = CurrencyCode
    posting_timestamp: datetime
    settlement_status: SettlementStatus
    reference_id: str | None = None


class Discrepancy(BaseModel):
    field: str
    expected: str | None
    actual: str | None
    difference: str | None = None
    severity: Severity
    rule_id: str
    explanation: str


class ReconciliationResult(BaseModel):
    """Output of the reconciliation engine (Phase 6). Source records are never modified."""

    transaction_id: str
    ledger_id: str | None
    status: Literal["reconciled", "discrepancy"]
    discrepancies: list[Discrepancy] = Field(default_factory=list)


class Evidence(BaseModel):
    evidence_id: str
    source: EvidenceSource
    description: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class Case(BaseModel):
    case_id: str
    customer_id: str
    transaction_ids: list[str] = Field(default_factory=list)
    status: CaseStatus = CaseStatus.CASE_CREATED
    evidence: list[Evidence] = Field(default_factory=list)
    decision: Decision | None = None
    reviewer: str | None = None
    created_at: datetime
    is_synthetic: Literal[True] = True
