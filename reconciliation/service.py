"""Reconciliation service: transaction records vs ledger records -> reconciled | discrepancy."""

import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime

from backend.app.schemas.domain import LedgerRecord, ReconciliationResult, Transaction
from data_pipeline.consolidation.models import CanonicalCase
from reconciliation.config import ReconciliationConfig
from reconciliation.discrepancy_detector import DiscrepancyDetector
from reconciliation.matcher import match_records

logger = logging.getLogger(__name__)


class ReconciliationService:
    """Read-only: inputs are never modified and results are new objects."""

    def __init__(self, config: ReconciliationConfig | None = None) -> None:
        self.detector = DiscrepancyDetector(config)

    def reconcile(
        self,
        transaction: Transaction,
        ledger_records: Sequence[LedgerRecord],
        as_of: datetime | None = None,
    ) -> ReconciliationResult:
        matched = match_records(transaction, ledger_records)
        discrepancies = self.detector.detect(matched, as_of)
        return ReconciliationResult(
            transaction_id=transaction.transaction_id,
            ledger_id=matched.primary.ledger_id if matched.primary else None,
            status="discrepancy" if discrepancies else "reconciled",
            discrepancies=discrepancies,
        )

    def reconcile_many(
        self,
        transactions: Iterable[Transaction],
        ledger_by_transaction: Mapping[str, Sequence[LedgerRecord]],
        as_of: datetime | None = None,
    ) -> list[ReconciliationResult]:
        results = [
            self.reconcile(tx, ledger_by_transaction.get(tx.transaction_id, ()), as_of)
            for tx in transactions
        ]
        results.sort(key=lambda r: r.transaction_id)
        counts = Counter(r.status for r in results)
        logger.info("reconciled %d transactions: %s", len(results), dict(counts))
        return results

    def reconcile_case(self, case: CanonicalCase) -> list[ReconciliationResult]:
        """Reconcile every transaction in a case (focus + context); as_of is the case time."""
        canonical = [*case.focus_transactions, *case.context_transactions]
        results = [self.reconcile(c.transaction, c.ledger_records, case.as_of) for c in canonical]
        results.sort(key=lambda r: r.transaction_id)
        return results
