"""In-memory consolidated view of all cleaned, normalized records with case construction."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta

from backend.app.schemas.domain import LedgerRecord, Transaction
from data_pipeline.consolidation.models import (
    CanonicalCase,
    CanonicalCustomer,
    CanonicalKYCDocument,
    CanonicalTransaction,
)


@dataclass
class ConsolidatedStore:
    customers: dict[str, CanonicalCustomer] = field(default_factory=dict)
    kyc_by_customer: dict[str, list[CanonicalKYCDocument]] = field(default_factory=dict)
    transactions_by_customer: dict[str, list[Transaction]] = field(default_factory=dict)
    ledger_by_transaction: dict[str, list[LedgerRecord]] = field(default_factory=dict)

    def _canonical_tx(self, tx: Transaction) -> CanonicalTransaction:
        return CanonicalTransaction(
            transaction=tx, ledger_records=self.ledger_by_transaction.get(tx.transaction_id, [])
        )

    def build_case(
        self,
        customer_id: str,
        focus_transaction_ids: list[str] | None = None,
        context_days: int = 30,
        case_id: str | None = None,
    ) -> CanonicalCase:
        """Build the case for a customer.

        Focus defaults to the customer's latest transaction. ``as_of`` is the latest focus
        timestamp; context holds the customer's other transactions in
        ``[as_of - context_days, as_of]`` so no future information can enter the case.
        """
        if customer_id not in self.customers:
            raise KeyError(f"unknown customer {customer_id}")
        history = self.transactions_by_customer.get(customer_id, [])
        by_id = {t.transaction_id: t for t in history}
        if focus_transaction_ids is None:
            if not history:
                raise ValueError(f"customer {customer_id} has no transactions to focus on")
            focus_transaction_ids = [history[-1].transaction_id]
        missing = [t for t in focus_transaction_ids if t not in by_id]
        if missing:
            raise ValueError(f"transactions not found for customer {customer_id}: {missing}")
        focus = [by_id[t] for t in focus_transaction_ids]
        as_of = max(t.timestamp for t in focus)
        window_start = as_of - timedelta(days=context_days)
        focus_ids = set(focus_transaction_ids)
        context = [
            t
            for t in history
            if t.transaction_id not in focus_ids and window_start <= t.timestamp <= as_of
        ]
        return CanonicalCase(
            case_id=case_id or f"CASE-{customer_id}-{focus_transaction_ids[0]}",
            customer=self.customers[customer_id],
            kyc_documents=self.kyc_by_customer.get(customer_id, []),
            focus_transactions=[
                self._canonical_tx(t) for t in sorted(focus, key=lambda t: t.timestamp)
            ],
            context_transactions=[self._canonical_tx(t) for t in context],
            as_of=as_of,
            context_days=context_days,
        )


def group_sorted(transactions: list[Transaction]) -> dict[str, list[Transaction]]:
    grouped: dict[str, list[Transaction]] = defaultdict(list)
    for tx in transactions:
        grouped[tx.customer_id].append(tx)
    for items in grouped.values():
        items.sort(key=lambda t: (t.timestamp, t.transaction_id))
    return dict(grouped)
