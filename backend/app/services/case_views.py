"""Compact, UI-oriented views built from a finished case (timeline, customer panel)."""

from typing import Any

from agents.state import CaseState


def build_timeline(state: CaseState) -> list[dict[str, Any]]:
    """Focus and context transactions, with reconciliation status and the focus anomaly score."""
    case = state.canonical_case
    if case is None:
        return []
    recon = {r.transaction_id: r for r in state.reconciliation_results}
    anomaly = {a.transaction_id: a for a in state.anomaly_findings}
    focus = {c.transaction.transaction_id for c in case.focus_transactions}
    items: list[dict[str, Any]] = []
    for canonical in [*case.context_transactions, *case.focus_transactions]:
        tx = canonical.transaction
        result = recon.get(tx.transaction_id)
        finding = anomaly.get(tx.transaction_id)
        items.append(
            {
                "transaction_id": tx.transaction_id,
                "timestamp": tx.timestamp.isoformat(),
                "amount": tx.amount,
                "currency": tx.currency,
                "type": tx.transaction_type.value,
                "channel": tx.channel.value,
                "sender_country": tx.sender_country,
                "receiver_country": tx.receiver_country,
                "is_focus": tx.transaction_id in focus,
                "reconciliation": result.status if result else None,
                "reconciliation_rules": [d.rule_id for d in result.discrepancies] if result else [],
                "anomaly_probability": finding.probability if finding else None,
                "anomaly_flagged": finding.flagged if finding else None,
            }
        )
    items.sort(key=lambda i: str(i["timestamp"]))
    return items


def build_customer(state: CaseState) -> dict[str, Any]:
    """Non-identifying summary plus a separate ``details`` block (personal data, analysts only)."""
    case = state.canonical_case
    if case is None:
        return {}
    c = case.customer.customer
    return {
        "customer_id": c.customer_id,
        "kyc_status": c.kyc_status.value,
        "account_type": c.account_type.value,
        "account_age_days": c.account_age_days,
        "country": c.country,
        "documents": [
            {
                "document_id": d.record.document_id,
                "document_type": d.record.document_type.value,
                "issue_date": d.record.issue_date.isoformat(),
                "expiry_date": d.record.expiry_date.isoformat(),
            }
            for d in case.kyc_documents
        ],
        "details": {
            "name": c.name,
            "alternate_names": c.alternate_names,
            "date_of_birth": c.date_of_birth.isoformat(),
            "address": c.address,
            "occupation": c.occupation,
            "document_names": {d.record.document_id: d.record.name for d in case.kyc_documents},
            "document_addresses": {
                d.record.document_id: d.record.address for d in case.kyc_documents
            },
            "document_dates_of_birth": {
                d.record.document_id: d.record.date_of_birth.isoformat() for d in case.kyc_documents
            },
        },
    }
