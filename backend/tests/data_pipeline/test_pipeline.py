import json
from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.ingestion.loader import IntegrityError, verify_manifest
from data_pipeline.pipeline import PipelineResult, run_pipeline, write_processed
from data_pipeline.synthetic.config import GeneratorConfig
from data_pipeline.synthetic.generate import SyntheticDataset, generate_dataset, write_dataset
from data_pipeline.synthetic.validation import ValidationReport

CFG = GeneratorConfig(seed=11, n_customers=300, n_transactions=6000)
LABEL_KEYS = {
    "is_anomaly",
    "anomaly_type",
    "episode_id",
    "discrepancy_type",
    "variation_type",
    "entity_id",
}


@pytest.fixture(scope="module")
def dataset() -> SyntheticDataset:
    return generate_dataset(CFG)


@pytest.fixture(scope="module")
def clean_dir(dataset: SyntheticDataset, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clean")
    write_dataset(dataset, path)
    return path


@pytest.fixture(scope="module")
def clean_result(clean_dir: Path) -> PipelineResult:
    return run_pipeline(clean_dir)


def _write_mutated(dataset: SyntheticDataset, path: Path, tables: dict[str, pd.DataFrame]) -> Path:
    write_dataset(SyntheticDataset(tables=tables, manifest={}, validation=ValidationReport()), path)
    return path


def test_clean_data_loads_fully_with_nothing_quarantined(
    clean_result: PipelineResult, dataset: SyntheticDataset
) -> None:
    r = clean_result.report
    assert r.manifest_verified is True
    assert not clean_result.quarantine
    assert r.accepted["customers"] == 300
    assert r.accepted["transactions"] == 6000
    assert r.accepted["ledger"] == len(dataset.tables["ledger"])
    assert r.addresses_unparsed == 0


def test_ledger_facts_match_generator_ground_truth(
    clean_result: PipelineResult, dataset: SyntheticDataset
) -> None:
    labels = dataset.tables["ledger_labels"]["discrepancy_type"].value_counts()
    assert clean_result.report.transactions_without_ledger == labels["missing_ledger_entry"]
    assert clean_result.report.transactions_with_multiple_ledger == labels["duplicate_posting"]
    # blank reference_id cells are the only null-token conversions in ledger.csv
    assert (
        clean_result.report.cleaning_actions["ledger"]["null_token_to_none"]
        == labels["missing_reference"]
    )


def test_tampered_file_fails_integrity_check(dataset: SyntheticDataset, tmp_path: Path) -> None:
    write_dataset(dataset, tmp_path)
    assert verify_manifest(tmp_path) is True
    with (tmp_path / "customers.csv").open("a") as handle:
        handle.write("tampered\n")
    with pytest.raises(IntegrityError):
        run_pipeline(tmp_path)


def test_dirty_rows_are_quarantined_with_reasons_not_dropped(
    dataset: SyntheticDataset, tmp_path: Path
) -> None:
    t = {k: v.copy() for k, v in dataset.tables.items()}
    tx = t["transactions"]
    orphan = tx.iloc[[0]].copy()
    orphan["transaction_id"] = "TXN-ORPHAN"
    orphan["customer_id"] = "CUST-NOPE"
    negative = tx.iloc[[1]].copy()
    negative["transaction_id"] = "TXN-NEG"
    negative["amount"] = -5.0
    bad_enum = tx.iloc[[2]].copy()
    bad_enum["transaction_id"] = "TXN-ENUM"
    bad_enum["channel"] = "carrier_pigeon"
    conflict = tx.iloc[[3]].copy()
    conflict["amount"] = conflict["amount"] + 1  # same id, different content
    exact_dup = tx.iloc[[4]].copy()
    t["transactions"] = pd.concat(
        [tx, orphan, negative, bad_enum, conflict, exact_dup], ignore_index=True
    )
    ledger = t["ledger"]
    stray = ledger.iloc[[0]].copy()
    stray["ledger_id"] = "LED-STRAY"
    stray["transaction_id"] = "TXN-GHOST"
    t["ledger"] = pd.concat([ledger, stray], ignore_index=True)

    result = run_pipeline(_write_mutated(dataset, tmp_path, t))
    by_reason = result.report.quarantined_by_reason
    assert by_reason == {
        "orphan_reference": 2,
        "schema_validation_failed": 2,
        "duplicate_key_conflict": 1,
    }
    assert result.report.exact_duplicates_dropped == {"transactions": 1}
    assert result.report.accepted["transactions"] == 6000  # originals untouched
    details = {
        q.detail.split(":")[0] for q in result.quarantine if q.reason == "schema_validation_failed"
    }
    assert details == {"amount", "channel"}


def test_messy_text_is_cleaned_but_raw_evidence_stays_recoverable(
    dataset: SyntheticDataset, tmp_path: Path
) -> None:
    t = {k: v.copy() for k, v in dataset.tables.items()}
    cid = t["customers"].loc[0, "customer_id"]
    t["customers"].loc[0, "name"] = "  ADAMS,   Ava  "
    t["customers"].loc[0, "address"] = "12 Oak  Street, Springfield, us"
    result = run_pipeline(_write_mutated(dataset, tmp_path, t))
    c = result.store.customers[cid]
    assert c.customer.name == "ADAMS, Ava"  # whitespace-cleaned, otherwise as ingested
    assert c.name.canonical == "ava adams" and c.name.reordered
    assert c.address.canonical == "12 oak st, springfield, US"
    assert result.report.cleaning_actions["customers"]["trim_whitespace"] >= 1


def test_unusable_name_is_quarantined(dataset: SyntheticDataset, tmp_path: Path) -> None:
    t = {k: v.copy() for k, v in dataset.tables.items()}
    t["customers"].loc[0, "name"] = "..."
    result = run_pipeline(_write_mutated(dataset, tmp_path, t))
    assert result.report.quarantined_by_reason["normalization_failed"] == 1
    # the customer's KYC record and transactions become orphans instead of vanishing
    assert result.report.quarantined_by_reason["orphan_reference"] >= 2


def test_case_has_no_future_data_and_no_labels(clean_result: PipelineResult) -> None:
    store = clean_result.store
    cid = max(store.transactions_by_customer, key=lambda c: len(store.transactions_by_customer[c]))
    history = store.transactions_by_customer[cid]
    focus = history[len(history) // 2]
    case = store.build_case(cid, [focus.transaction_id], context_days=30)
    assert case.as_of == focus.timestamp
    assert all(t.transaction.timestamp <= case.as_of for t in case.context_transactions)
    assert focus.transaction_id not in {
        t.transaction.transaction_id for t in case.context_transactions
    }
    assert any(t.timestamp > case.as_of for t in history)  # future existed but was excluded

    def keys(obj: object) -> set[str]:
        if isinstance(obj, dict):
            return set(obj) | {k for v in obj.values() for k in keys(v)}
        if isinstance(obj, list):
            return {k for v in obj for k in keys(v)}
        return set()

    assert not keys(json.loads(case.model_dump_json())) & LABEL_KEYS


def test_case_defaults_to_latest_transaction_and_attaches_ledger(
    clean_result: PipelineResult,
) -> None:
    store = clean_result.store
    cid = next(iter(store.transactions_by_customer))
    case = store.build_case(cid)
    latest = store.transactions_by_customer[cid][-1]
    assert case.focus_transactions[0].transaction.transaction_id == latest.transaction_id
    assert case.focus_transactions[0].ledger_records == store.ledger_by_transaction.get(
        latest.transaction_id, []
    )
    assert case.kyc_documents and case.case_id.startswith("CASE-")


def test_build_case_rejects_bad_input(clean_result: PipelineResult) -> None:
    store = clean_result.store
    cid, other = list(store.transactions_by_customer)[:2]
    with pytest.raises(KeyError):
        store.build_case("CUST-NOPE")
    with pytest.raises(ValueError):  # a transaction belonging to someone else
        store.build_case(cid, [store.transactions_by_customer[other][0].transaction_id])


def test_write_processed_outputs(clean_result: PipelineResult, tmp_path: Path) -> None:
    write_processed(clean_result, tmp_path)
    assert (
        json.loads((tmp_path / "pipeline_report.json").read_text())["accepted"]["customers"] == 300
    )
    kyc = pd.read_csv(tmp_path / "kyc_normalized.csv")
    assert {"name_raw", "name_canonical", "document_number"} <= set(kyc.columns)
    assert len(kyc) == 300
