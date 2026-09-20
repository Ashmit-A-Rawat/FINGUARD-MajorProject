"""Canonical, engine-ready representations. Raw records are embedded unchanged."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from backend.app.schemas.domain import Customer, KYCRecord, LedgerRecord, Transaction
from data_pipeline.normalization.models import NormalizedAddress, NormalizedName


class CanonicalCustomer(BaseModel):
    customer: Customer
    name: NormalizedName
    alternate_names: list[NormalizedName] = Field(default_factory=list)
    address: NormalizedAddress


class CanonicalKYCDocument(BaseModel):
    record: KYCRecord
    name: NormalizedName
    address: NormalizedAddress
    document_number: str  # normalized (upper-case alphanumeric)


class CanonicalTransaction(BaseModel):
    """A transaction with every ledger record observed for it (0, 1 or several).

    The number of ledger records is a *fact*; judging it is the reconciliation engine's job.
    """

    transaction: Transaction
    ledger_records: list[LedgerRecord] = Field(default_factory=list)


class CanonicalCase(BaseModel):
    """Input representation handed to the KYC, anomaly and reconciliation engines.

    Contains no ground-truth labels and no transaction later than ``as_of``.
    """

    case_id: str
    customer: CanonicalCustomer
    kyc_documents: list[CanonicalKYCDocument]
    focus_transactions: list[CanonicalTransaction]
    context_transactions: list[CanonicalTransaction]
    as_of: datetime
    context_days: int
    is_synthetic: Literal[True] = True
