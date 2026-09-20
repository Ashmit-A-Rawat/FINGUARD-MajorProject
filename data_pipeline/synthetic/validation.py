"""Dataset validation: schema conformance, uniqueness, referential integrity, label sanity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import BaseModel, ValidationError

from backend.app.schemas.domain import Customer, KYCRecord, LedgerRecord, Transaction
from data_pipeline.synthetic.config import (
    ANOMALY_TYPES,
    KYC_VARIATIONS,
    LEDGER_DISCREPANCIES,
    GeneratorConfig,
)

MAX_REPORTED_ERRORS = 20


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    checks_run: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def check(self, condition: bool, message: str) -> None:
        self.checks_run += 1
        if not condition:
            self.errors.append(message)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.astype(object).where(df.notna(), None).to_dict("records")  # type: ignore[no-any-return]


def _validate_rows(
    model: type[BaseModel], df: pd.DataFrame, name: str, report: ValidationReport
) -> None:
    failures = 0
    for i, row in enumerate(_records(df)):
        try:
            model.model_validate(row)
        except ValidationError as exc:
            failures += 1
            if failures <= MAX_REPORTED_ERRORS:
                report.errors.append(f"{name} row {i}: {exc.errors()[0]['msg']}")
    report.checks_run += 1
    if failures > MAX_REPORTED_ERRORS:
        report.errors.append(f"{name}: {failures} rows failed schema validation in total")


def validate_tables(tables: dict[str, pd.DataFrame], cfg: GeneratorConfig) -> ValidationReport:
    r = ValidationReport()
    cust, kyc, tx, led = (tables[k] for k in ("customers", "kyc_records", "transactions", "ledger"))

    for model, df, name in (
        (Customer, cust, "customers"),
        (KYCRecord, kyc, "kyc_records"),
        (Transaction, tx, "transactions"),
        (LedgerRecord, led, "ledger"),
    ):
        _validate_rows(model, df, name, r)

    r.check(cust["customer_id"].is_unique, "customer_id not unique")
    r.check(kyc["document_id"].is_unique, "document_id not unique")
    r.check(tx["transaction_id"].is_unique, "transaction_id not unique")
    r.check(led["ledger_id"].is_unique, "ledger_id not unique")
    r.check(len(cust) == cfg.n_customers, f"expected {cfg.n_customers} customers, got {len(cust)}")
    r.check(
        len(tx) == cfg.n_transactions, f"expected {cfg.n_transactions} transactions, got {len(tx)}"
    )

    customer_ids = set(cust["customer_id"])
    r.check(set(kyc["customer_id"]) <= customer_ids, "kyc_records reference unknown customers")
    r.check(set(tx["customer_id"]) <= customer_ids, "transactions reference unknown customers")
    r.check(
        set(led["transaction_id"]) <= set(tx["transaction_id"]),
        "ledger references unknown transactions",
    )

    start, end = pd.Timestamp(cfg.window_start), pd.Timestamp(cfg.window_end) + pd.Timedelta(days=1)
    ts = pd.to_datetime(tx["timestamp"])
    r.check(
        bool(((ts >= start) & (ts < end)).all()),
        "transaction timestamps outside observation window",
    )
    r.check(bool(tx["timestamp"].is_monotonic_increasing), "transactions not sorted by timestamp")

    labels = tables["transaction_labels"]
    r.check(
        set(labels["transaction_id"]) == set(tx["transaction_id"]),
        "transaction_labels ids mismatch",
    )
    r.check(
        set(labels["anomaly_type"]) <= set(ANOMALY_TYPES) | {"none"},
        "unknown anomaly type in labels",
    )
    ll = tables["ledger_labels"]
    r.check(
        set(ll["discrepancy_type"]) <= set(LEDGER_DISCREPANCIES), "unknown ledger discrepancy type"
    )
    r.check(
        set(ll["transaction_id"]) <= set(tx["transaction_id"]), "ledger_labels reference unknown tx"
    )
    kl = tables["kyc_labels"]
    r.check(set(kl["variation_type"]) <= set(KYC_VARIATIONS), "unknown KYC variation type")
    r.check(set(kl["document_id"]) == set(kyc["document_id"]), "kyc_labels ids mismatch")
    el = tables["entity_labels"]
    r.check(set(el["customer_id"]) == customer_ids, "entity_labels ids mismatch")
    return r
