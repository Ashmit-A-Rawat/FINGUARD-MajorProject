"""Governance-risk (the "G" in ESG) indicator, added in response to review feedback asking us to
explore an ESG score as an additional evidence parameter.

SCOPE, STATED HONESTLY: this implements only the Governance pillar, as two standard compliance
heuristics computed from data we already have. Environmental and Social scoring need external
sector, emissions or sustainability datasets this project does not have, so they are NOT
implemented; see the "Environmental and Social" note below for exactly what would be needed.

Two deterministic checks, each relative to the CUSTOMER'S OWN history (never a claim about any
real country or person):
  1. new_cross_border_counterparty: the transaction moves money to or from a country this customer
     has never transacted with before. Unexpected new cross-border counterparties are a standard
     enhanced-due-diligence trigger in real compliance practice.
  2. structuring_suspected: several of the customer's transactions, within a trailing window, sit
     just under a round reporting-style threshold -- the classic "structuring" / "smurfing" pattern
     used to avoid a reporting requirement.

Reuses the exact (transaction, history) shape and the counterparty/country helpers already used by
llm/rag/evidence.behaviour_evidence, so this is additive, not a parallel implementation.

LIMITATIONS (read before using a number from this module):
- The reporting threshold is a configurable, illustrative figure, not tied to any real
  jurisdiction's actual law.
- Amounts are compared in the transaction's own currency with no FX normalisation, so the
  structuring check is only meaningful within one currency's transactions; see GovernanceConfig.
- This is a standalone, tested proof of concept. It is wired into the evidence-validation /
  policy-floor layer (guardrails/policy_checks.py) and covered by tests, but it is NOT yet called
  by the live multi-agent workflow (agents/coordinator/workflow.py); see docs/architecture/
  esg-governance.md for exactly what remains to wire it in.

Environmental and Social: a real score would need, at minimum, a counterparty/industry-sector
classification (we do not have one; customer `occupation` in the synthetic data is a personal job
title, not a counterparty sector) and an external emissions or sustainability dataset. Until such
data exists, implementing E and S here would mean inventing numbers, which this project's own rule
("no fabricated results") rules out. This is recorded as the extension point, not filled in.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from backend.app.schemas.domain import Evidence, EvidenceSource, Transaction

OUTGOING = ("transfer_out", "payment", "withdrawal")


class GovernanceConfig(BaseModel):
    structuring_window_days: int = 30
    structuring_threshold_amount: float = Field(10_000.0, gt=0)
    structuring_band_fraction: float = Field(0.9, gt=0, lt=1)  # lower edge of the "just under" band
    structuring_min_count: int = Field(3, ge=2)


def _counterparty(t: Transaction) -> str:
    return t.receiver if t.transaction_type.value in OUTGOING else t.sender


def _country(t: Transaction) -> str:
    return t.receiver_country if t.transaction_type.value in OUTGOING else t.sender_country


def score_governance(
    tx: Transaction, history: Sequence[Transaction], config: GovernanceConfig | None = None
) -> dict[str, Any]:
    """Pure computation (no Evidence wrapping), so it is easy to unit-test and reuse."""
    cfg = config or GovernanceConfig()
    prior = [t for t in history if t.timestamp < tx.timestamp]

    new_country = not any(_country(t) == _country(tx) for t in prior)
    cross_border = tx.sender_country != tx.receiver_country

    window_start = tx.timestamp - timedelta(days=cfg.structuring_window_days)
    lower = cfg.structuring_threshold_amount * cfg.structuring_band_fraction
    in_band = [
        t
        for t in [*prior, tx]
        if t.currency == tx.currency
        and window_start <= t.timestamp <= tx.timestamp
        and lower <= t.amount < cfg.structuring_threshold_amount
    ]
    structuring = len(in_band) >= cfg.structuring_min_count

    return {
        "new_cross_border_counterparty": bool(new_country and cross_border),
        "counterparty_country": _country(tx),
        "structuring_suspected": structuring,
        "structuring_window_days": cfg.structuring_window_days,
        "structuring_threshold_amount": cfg.structuring_threshold_amount,
        "sub_threshold_transactions_in_window": len(in_band),
        "flagged": bool(new_country and cross_border) or structuring,
    }


def governance_evidence(
    tx: Transaction,
    history: Sequence[Transaction],
    created_at: datetime,
    config: GovernanceConfig | None = None,
) -> Evidence:
    facts = score_governance(tx, history, config)
    reasons = []
    if facts["new_cross_border_counterparty"]:
        reasons.append(
            f"first transaction with counterparty country {facts['counterparty_country']}"
        )
    if facts["structuring_suspected"]:
        reasons.append(
            f"{facts['sub_threshold_transactions_in_window']} transactions just under "
            f"{facts['structuring_threshold_amount']:.0f} {tx.currency} within "
            f"{facts['structuring_window_days']} days"
        )
    description = (
        "Governance indicators: " + "; ".join(reasons)
        if reasons
        else "Governance indicators: none triggered"
    )
    return Evidence(
        evidence_id=f"ESG-{tx.transaction_id}",
        source=EvidenceSource.ESG,
        description=description,
        payload=facts,
        created_at=created_at,
    )
