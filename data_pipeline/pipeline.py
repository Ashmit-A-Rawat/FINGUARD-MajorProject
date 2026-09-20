"""End-to-end pipeline: ingest -> clean -> validate -> normalize -> consolidate.

Every stage reports what it did; rejected rows are quarantined with a reason.
"""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from backend.app.schemas.domain import Customer, KYCRecord, LedgerRecord, Transaction
from data_pipeline.cleaning.cleaner import clean_record
from data_pipeline.consolidation.canonicalize import canonicalize_customer, canonicalize_kyc
from data_pipeline.consolidation.models import CanonicalKYCDocument
from data_pipeline.consolidation.store import ConsolidatedStore, group_sorted
from data_pipeline.ingestion.loader import TABLE_FILES, read_raw_table, verify_manifest
from data_pipeline.ingestion.validator import Quarantined, validate_records

logger = logging.getLogger(__name__)

MODELS: dict[str, type[BaseModel]] = {
    "customers": Customer,
    "kyc_records": KYCRecord,
    "transactions": Transaction,
    "ledger": LedgerRecord,
}


class PipelineReport(BaseModel):
    source_dir: str
    manifest_verified: bool | None
    rows_read: dict[str, int]
    accepted: dict[str, int]
    exact_duplicates_dropped: dict[str, int]
    quarantined_by_reason: dict[str, int]
    cleaning_actions: dict[str, dict[str, int]]
    addresses_unparsed: int
    transactions_without_ledger: int
    transactions_with_multiple_ledger: int


@dataclass
class PipelineResult:
    store: ConsolidatedStore
    report: PipelineReport
    quarantine: list[Quarantined]


def _quarantine(
    table: str, number: int, reason: str, detail: str, record: BaseModel
) -> Quarantined:
    return Quarantined(
        table=table,
        row_number=number,
        reason=reason,
        detail=detail,
        record=json.loads(record.model_dump_json()),
    )


def run_pipeline(input_dir: Path) -> PipelineResult:
    verified = verify_manifest(input_dir)
    cleaning: dict[str, Counter[str]] = {}
    duplicates: Counter[str] = Counter()
    rows_read: dict[str, int] = {}
    quarantine: list[Quarantined] = []
    accepted: dict[str, list[BaseModel]] = {}

    for table, filename in TABLE_FILES.items():
        raw = read_raw_table(input_dir / filename)
        rows_read[table] = len(raw)
        actions: Counter[str] = Counter()
        cleaned = [clean_record(row, actions) for row in raw]
        cleaning[table] = actions
        valid, rejected = validate_records(MODELS[table], table, cleaned, duplicates)
        accepted[table], quarantine = valid, quarantine + rejected
        logger.info(
            "%s: read=%d accepted=%d rejected=%d", table, len(raw), len(valid), len(rejected)
        )

    customers = {c.customer_id: c for c in accepted["customers"] if isinstance(c, Customer)}
    store = ConsolidatedStore()
    unparsed = 0

    for customer in customers.values():
        try:
            canonical = canonicalize_customer(customer)
        except ValueError as exc:
            quarantine.append(
                _quarantine("customers", 0, "normalization_failed", str(exc), customer)
            )
            continue
        unparsed += not canonical.address.parsed
        store.customers[customer.customer_id] = canonical

    kyc_docs: dict[str, list[CanonicalKYCDocument]] = defaultdict(list)
    for record in accepted["kyc_records"]:
        assert isinstance(record, KYCRecord)
        if record.customer_id not in store.customers:
            quarantine.append(
                _quarantine(
                    "kyc_records",
                    0,
                    "orphan_reference",
                    f"unknown customer {record.customer_id}",
                    record,
                )
            )
            continue
        try:
            document = canonicalize_kyc(record)
        except ValueError as exc:
            quarantine.append(
                _quarantine("kyc_records", 0, "normalization_failed", str(exc), record)
            )
            continue
        unparsed += not document.address.parsed
        kyc_docs[record.customer_id].append(document)
    store.kyc_by_customer = dict(kyc_docs)

    transactions: list[Transaction] = []
    for tx in accepted["transactions"]:
        assert isinstance(tx, Transaction)
        if tx.customer_id in store.customers:
            transactions.append(tx)
        else:
            quarantine.append(
                _quarantine(
                    "transactions", 0, "orphan_reference", f"unknown customer {tx.customer_id}", tx
                )
            )
    store.transactions_by_customer = group_sorted(transactions)

    known_tx = {t.transaction_id for t in transactions}
    ledger: dict[str, list[LedgerRecord]] = defaultdict(list)
    for entry in accepted["ledger"]:
        assert isinstance(entry, LedgerRecord)
        if entry.transaction_id in known_tx:
            ledger[entry.transaction_id].append(entry)
        else:
            quarantine.append(
                _quarantine(
                    "ledger",
                    0,
                    "orphan_reference",
                    f"unknown transaction {entry.transaction_id}",
                    entry,
                )
            )
    store.ledger_by_transaction = dict(ledger)

    report = PipelineReport(
        source_dir=str(input_dir),
        manifest_verified=verified,
        rows_read=rows_read,
        accepted={
            "customers": len(store.customers),
            "kyc_records": sum(len(v) for v in store.kyc_by_customer.values()),
            "transactions": len(transactions),
            "ledger": sum(len(v) for v in ledger.values()),
        },
        exact_duplicates_dropped=dict(duplicates),
        quarantined_by_reason=dict(Counter(q.reason for q in quarantine)),
        cleaning_actions={t: dict(c) for t, c in cleaning.items()},
        addresses_unparsed=unparsed,
        transactions_without_ledger=len(known_tx - set(ledger)),
        transactions_with_multiple_ledger=sum(1 for v in ledger.values() if len(v) > 1),
    )
    return PipelineResult(store=store, report=report, quarantine=quarantine)


def write_processed(result: PipelineResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pipeline_report.json").write_text(result.report.model_dump_json(indent=2))
    (out_dir / "quarantine.json").write_text(
        json.dumps([q.model_dump(mode="json") for q in result.quarantine], indent=2)
    )
    pd.DataFrame(
        [
            {
                "customer_id": c.customer.customer_id,
                "name_raw": c.name.raw,
                "name_canonical": c.name.canonical,
                "address_raw": c.address.raw,
                "address_canonical": c.address.canonical,
            }
            for c in result.store.customers.values()
        ]
    ).to_csv(out_dir / "customers_normalized.csv", index=False)
    pd.DataFrame(
        [
            {
                "document_id": d.record.document_id,
                "customer_id": d.record.customer_id,
                "name_raw": d.name.raw,
                "name_canonical": d.name.canonical,
                "address_raw": d.address.raw,
                "address_canonical": d.address.canonical,
                "document_number": d.document_number,
            }
            for docs in result.store.kyc_by_customer.values()
            for d in docs
        ]
    ).to_csv(out_dir / "kyc_normalized.csv", index=False)
