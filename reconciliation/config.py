"""Reconciliation tolerances. These are business parameters, not learned values."""

from datetime import timedelta

from pydantic import BaseModel, ConfigDict, Field


class ReconciliationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Amounts are compared in whole cents. A difference up to this many cents is tolerated.
    amount_tolerance_cents: int = Field(1, ge=0)
    amount_high_severity_pct: float = Field(0.10, gt=0)  # >= 10% of the transaction amount
    amount_medium_severity_pct: float = Field(0.01, gt=0)  # >= 1%; below that: low

    # Settlement convention: posting later than this after the transaction is late (T+3).
    max_posting_lag: timedelta = timedelta(days=3)
    late_high_severity_multiple: float = Field(3.0, gt=1)  # lag > 3x the limit: high

    # Settlement-status checks (ledger status vs the expected "settled").
    check_settlement_status: bool = True
    # Pending longer than this after the transaction is stale (needs an `as_of` time).
    stale_pending_after: timedelta = timedelta(days=3)
