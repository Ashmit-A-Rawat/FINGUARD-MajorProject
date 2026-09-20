"""Pair a transaction with its ledger record(s). Never modifies either."""

from collections.abc import Sequence
from dataclasses import dataclass

from backend.app.schemas.domain import LedgerRecord, Transaction


@dataclass(frozen=True)
class MatchedRecords:
    transaction: Transaction
    primary: LedgerRecord | None  # earliest posting; None when the ledger has no entry
    extras: tuple[LedgerRecord, ...]  # further ledger records for the same transaction


def match_records(
    transaction: Transaction, ledger_records: Sequence[LedgerRecord]
) -> MatchedRecords:
    """Match by transaction_id. With several candidates the earliest posting is primary
    (ties broken by ledger_id) so the choice is deterministic and the rest are 'extras'."""
    candidates = sorted(
        (r for r in ledger_records if r.transaction_id == transaction.transaction_id),
        key=lambda r: (r.posting_timestamp, r.ledger_id),
    )
    if not candidates:
        return MatchedRecords(transaction, None, ())
    return MatchedRecords(transaction, candidates[0], tuple(candidates[1:]))
