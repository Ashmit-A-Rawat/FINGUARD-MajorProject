"""Reconciliation rules. Each rule has a stable id, checks one thing, and explains itself.

Fields compared: transaction id (matching), ledger presence/count, amount, currency, reference id,
posting timestamp, settlement status. NOT compared: sender / receiver, because the ledger
schema carries no counterparty fields (a schema change would be needed to add them).
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from backend.app.schemas.domain import Discrepancy, Severity
from reconciliation.config import ReconciliationConfig
from reconciliation.matcher import MatchedRecords


@dataclass(frozen=True)
class Rule:
    rule_id: str
    name: str
    group: str  # "core" (data integrity) or "status" (settlement-status observations)
    check: Callable[[MatchedRecords, ReconciliationConfig, datetime | None], Discrepancy | None]


def _cents(value: float) -> int:
    return round(value * 100)


def _missing_ledger(
    m: MatchedRecords, c: ReconciliationConfig, _: datetime | None
) -> Discrepancy | None:
    if m.primary is not None:
        return None
    return Discrepancy(
        field="ledger_record",
        expected="1 ledger entry",
        actual="none",
        difference="-1",
        severity=Severity.HIGH,
        rule_id="REC-001",
        explanation=f"No ledger entry exists for transaction {m.transaction.transaction_id}.",
    )


def _duplicate_posting(
    m: MatchedRecords, c: ReconciliationConfig, _: datetime | None
) -> Discrepancy | None:
    if not m.extras:
        return None
    ids = ", ".join(r.ledger_id for r in (m.primary, *m.extras) if r is not None)
    return Discrepancy(
        field="ledger_record_count",
        expected="1",
        actual=str(1 + len(m.extras)),
        difference=f"+{len(m.extras)}",
        severity=Severity.HIGH,
        rule_id="REC-002",
        explanation=f"Posted {1 + len(m.extras)} times ({ids}); risk of double posting.",
    )


def _amount(m: MatchedRecords, c: ReconciliationConfig, _: datetime | None) -> Discrepancy | None:
    if m.primary is None:
        return None
    expected, actual = m.transaction.amount, m.primary.posted_amount
    diff_cents = _cents(actual) - _cents(expected)
    if abs(diff_cents) <= c.amount_tolerance_cents:
        return None
    pct = abs(diff_cents) / max(abs(_cents(expected)), 1)
    severity = (
        Severity.HIGH
        if pct >= c.amount_high_severity_pct
        else Severity.MEDIUM
        if pct >= c.amount_medium_severity_pct
        else Severity.LOW
    )
    return Discrepancy(
        field="amount",
        expected=f"{expected:.2f}",
        actual=f"{actual:.2f}",
        difference=f"{diff_cents / 100:+.2f} ({'+' if diff_cents > 0 else '-'}{pct:.1%})",
        severity=severity,
        rule_id="REC-003",
        explanation=(
            f"Ledger posted {actual:.2f} but the transaction amount is {expected:.2f} "
            f"(tolerance {c.amount_tolerance_cents} cent)."
        ),
    )


def _currency(m: MatchedRecords, c: ReconciliationConfig, _: datetime | None) -> Discrepancy | None:
    if m.primary is None or m.primary.currency == m.transaction.currency:
        return None
    return Discrepancy(
        field="currency",
        expected=m.transaction.currency,
        actual=m.primary.currency,
        difference=f"{m.transaction.currency} -> {m.primary.currency}",
        severity=Severity.HIGH,
        rule_id="REC-004",
        explanation="Ledger currency differs from the transaction currency.",
    )


def _reference_missing(
    m: MatchedRecords, c: ReconciliationConfig, _: datetime | None
) -> Discrepancy | None:
    if m.primary is None:
        return None
    tx_ref, led_ref = m.transaction.reference_id, m.primary.reference_id
    if (tx_ref is None) == (led_ref is None):
        return None
    side = "ledger record" if led_ref is None else "transaction"
    return Discrepancy(
        field="reference_id",
        expected=tx_ref,
        actual=led_ref,
        difference="missing on one side",
        severity=Severity.MEDIUM,
        rule_id="REC-005",
        explanation=f"The {side} has no reference id; the records cannot be tied by reference.",
    )


def _reference_mismatch(
    m: MatchedRecords, c: ReconciliationConfig, _: datetime | None
) -> Discrepancy | None:
    if m.primary is None:
        return None
    tx_ref, led_ref = m.transaction.reference_id, m.primary.reference_id
    if tx_ref is None or led_ref is None or tx_ref == led_ref:
        return None
    return Discrepancy(
        field="reference_id",
        expected=tx_ref,
        actual=led_ref,
        difference="different values",
        severity=Severity.HIGH,
        rule_id="REC-006",
        explanation="Both records carry a reference id but the values differ.",
    )


def _posting_before_transaction(
    m: MatchedRecords, c: ReconciliationConfig, _: datetime | None
) -> Discrepancy | None:
    if m.primary is None or m.primary.posting_timestamp >= m.transaction.timestamp:
        return None
    lag = m.primary.posting_timestamp - m.transaction.timestamp
    return Discrepancy(
        field="posting_timestamp",
        expected=f">= {m.transaction.timestamp.isoformat()}",
        actual=m.primary.posting_timestamp.isoformat(),
        difference=str(lag),
        severity=Severity.HIGH,
        rule_id="REC-007",
        explanation="The ledger posting is timestamped before the transaction occurred.",
    )


def _late_posting(
    m: MatchedRecords, c: ReconciliationConfig, _: datetime | None
) -> Discrepancy | None:
    if m.primary is None:
        return None
    lag = m.primary.posting_timestamp - m.transaction.timestamp
    if lag <= c.max_posting_lag:
        return None
    severity = (
        Severity.HIGH
        if lag > c.max_posting_lag * c.late_high_severity_multiple
        else Severity.MEDIUM
    )
    return Discrepancy(
        field="posting_timestamp",
        expected=f"within {c.max_posting_lag} of the transaction",
        actual=m.primary.posting_timestamp.isoformat(),
        difference=f"+{lag - c.max_posting_lag} over limit",
        severity=severity,
        rule_id="REC-008",
        explanation=f"Ledger posted {lag} after the transaction; the limit is {c.max_posting_lag}.",
    )


def _settlement_failed(
    m: MatchedRecords, c: ReconciliationConfig, _: datetime | None
) -> Discrepancy | None:
    if (
        not c.check_settlement_status
        or m.primary is None
        or m.primary.settlement_status != "failed"
    ):
        return None
    return Discrepancy(
        field="settlement_status",
        expected="settled",
        actual="failed",
        difference="failed settlement",
        severity=Severity.MEDIUM,
        rule_id="REC-009",
        explanation="The ledger records a failed settlement for an existing transaction.",
    )


def _stale_pending(
    m: MatchedRecords, c: ReconciliationConfig, as_of: datetime | None
) -> Discrepancy | None:
    if not c.check_settlement_status or m.primary is None or as_of is None:
        return None
    if m.primary.settlement_status != "pending":
        return None
    age = as_of - m.transaction.timestamp
    if age <= c.stale_pending_after:
        return None
    return Discrepancy(
        field="settlement_status",
        expected="settled",
        actual="pending",
        difference=f"pending for {age}",
        severity=Severity.MEDIUM,
        rule_id="REC-010",
        explanation=f"Still pending {age} after the transaction (limit {c.stale_pending_after}).",
    )


RULES: tuple[Rule, ...] = (
    Rule("REC-001", "missing_ledger_entry", "core", _missing_ledger),
    Rule("REC-002", "duplicate_posting", "core", _duplicate_posting),
    Rule("REC-003", "amount_mismatch", "core", _amount),
    Rule("REC-004", "currency_mismatch", "core", _currency),
    Rule("REC-005", "reference_missing", "core", _reference_missing),
    Rule("REC-006", "reference_mismatch", "core", _reference_mismatch),
    Rule("REC-007", "posting_before_transaction", "core", _posting_before_transaction),
    Rule("REC-008", "late_posting", "core", _late_posting),
    Rule("REC-009", "settlement_failed", "status", _settlement_failed),
    Rule("REC-010", "stale_pending", "status", _stale_pending),
)
