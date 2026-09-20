"""Turn engine outputs into Evidence objects for the investigation prompt.

Evidence carries identifiers and facts, never customer names or full document numbers (see the
data-privacy policy). Ids are stable and predictable so reports can cite them.
"""

from datetime import datetime
from statistics import median

from backend.app.schemas.domain import (
    Customer,
    Evidence,
    EvidenceSource,
    ReconciliationResult,
    Transaction,
)

OUTGOING = ("transfer_out", "payment", "withdrawal")


def transaction_evidence(
    tx: Transaction, created_at: datetime, memo: str | None = None
) -> Evidence:
    payload: dict[str, object] = {
        "transaction_id": tx.transaction_id,
        "customer_id": tx.customer_id,
        "timestamp": tx.timestamp.isoformat(),
        "amount": tx.amount,
        "currency": tx.currency,
        "type": tx.transaction_type.value,
        "channel": tx.channel.value,
        "sender_country": tx.sender_country,
        "receiver_country": tx.receiver_country,
    }
    if memo is not None:  # free text supplied by a counterparty: untrusted content
        payload["payment_memo"] = memo
    return Evidence(
        evidence_id=tx.transaction_id,
        source=EvidenceSource.TRANSACTION,
        description=(
            f"{tx.transaction_type.value} of {tx.amount:.2f} {tx.currency} via {tx.channel.value}"
        ),
        payload=payload,
        created_at=created_at,
    )


def customer_evidence(customer: Customer, created_at: datetime) -> Evidence:
    return Evidence(
        evidence_id=f"KYC-{customer.customer_id}",
        source=EvidenceSource.KYC,
        description=(
            f"KYC status is {customer.kyc_status.value}; account type {customer.account_type.value}"
        ),
        payload={
            "customer_id": customer.customer_id,
            "kyc_status": customer.kyc_status.value,
            "account_type": customer.account_type.value,
            "account_age_days": customer.account_age_days,
        },
        created_at=created_at,
    )


def reconciliation_evidence(result: ReconciliationResult, created_at: datetime) -> list[Evidence]:
    """One evidence item per discrepancy, or a single 'reconciled' item."""
    if not result.discrepancies:
        return [
            Evidence(
                evidence_id=f"REC-{result.transaction_id}-OK",
                source=EvidenceSource.RECONCILIATION,
                description="Transaction and ledger record agree on all compared fields",
                payload={"status": "reconciled", "ledger_id": result.ledger_id},
                created_at=created_at,
            )
        ]
    return [
        Evidence(
            evidence_id=f"REC-{result.transaction_id}-{d.rule_id}",
            source=EvidenceSource.RECONCILIATION,
            description=f"{d.rule_id} {d.field}: {d.explanation}",
            payload={
                "rule_id": d.rule_id,
                "field": d.field,
                "expected": d.expected,
                "actual": d.actual,
                "difference": d.difference,
                "severity": d.severity.value,
            },
            created_at=created_at,
        )
        for d in result.discrepancies
    ]


def behaviour_evidence(
    tx: Transaction, history: list[Transaction], created_at: datetime
) -> Evidence:
    """Plain, deterministic behavioural facts relative to the customer's PRIOR transactions only."""
    prior = [t for t in history if t.timestamp < tx.timestamp]
    ratio = None
    if prior:
        typical = median(t.amount for t in prior)
        ratio = round(tx.amount / typical, 2) if typical else None

    def within(seconds: int) -> int:
        return sum(1 for t in prior if (tx.timestamp - t.timestamp).total_seconds() <= seconds)

    def counterparty(t: Transaction) -> str:
        return t.receiver if t.transaction_type.value in OUTGOING else t.sender

    def country(t: Transaction) -> str:
        return t.receiver_country if t.transaction_type.value in OUTGOING else t.sender_country

    same_amount = sum(
        1 for t in prior if t.amount == tx.amount and counterparty(t) == counterparty(tx)
    )
    payload: dict[str, object] = {
        "prior_transactions": len(prior),
        "typical_prior_amount_ratio": ratio,
        "transactions_in_prior_10_min": within(600),
        "transactions_in_prior_24_h": within(86400),
        "hour_of_day": tx.timestamp.hour,
        "counterparty_seen_before": any(counterparty(t) == counterparty(tx) for t in prior),
        "counterparty_country_seen_before": any(country(t) == country(tx) for t in prior),
        "prior_identical_transfers_to_same_counterparty": same_amount,
    }
    return Evidence(
        evidence_id=f"BEH-{tx.transaction_id}",
        source=EvidenceSource.ANOMALY,
        description="Behaviour of this transaction compared with the customer's earlier activity",
        payload=payload,
        created_at=created_at,
    )
