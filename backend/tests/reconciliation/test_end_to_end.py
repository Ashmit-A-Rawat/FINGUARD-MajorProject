from pathlib import Path

import pytest

from data_pipeline.pipeline import run_pipeline
from data_pipeline.synthetic.config import GeneratorConfig
from data_pipeline.synthetic.generate import generate_dataset, write_dataset
from reconciliation.service import ReconciliationService

RULE_TO_LABEL = {
    "REC-001": "missing_ledger_entry",
    "REC-002": "duplicate_posting",
    "REC-003": "contradictory_amount",
    "REC-004": "currency_mismatch",
    "REC-005": "missing_reference",
    "REC-008": "late_posting",
}


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    path: Path = tmp_path_factory.mktemp("recon")
    dataset = generate_dataset(GeneratorConfig(seed=33, n_customers=300, n_transactions=8000))
    write_dataset(dataset, path)
    result = run_pipeline(path)
    txs = [t for v in result.store.transactions_by_customer.values() for t in v]
    as_of = max(t.timestamp for t in txs)
    results = ReconciliationService().reconcile_many(txs, result.store.ledger_by_transaction, as_of)
    return result, results, dataset.tables["ledger_labels"]


def test_core_rules_reproduce_injected_discrepancies_exactly(run) -> None:  # type: ignore[no-untyped-def]
    _, results, labels = run
    truth = dict(zip(labels["transaction_id"], labels["discrepancy_type"], strict=True))
    got = {
        r.transaction_id: {
            RULE_TO_LABEL[d.rule_id] for d in r.discrepancies if d.rule_id in RULE_TO_LABEL
        }
        for r in results
    }
    got = {t: kinds for t, kinds in got.items() if kinds}
    assert got == {t: {k} for t, k in truth.items()}


def test_no_core_false_positives_on_clean_transactions(run) -> None:  # type: ignore[no-untyped-def]
    _, results, labels = run
    labelled = set(labels["transaction_id"])
    core = {f"REC-00{i}" for i in range(1, 9)}
    clean = [r for r in results if r.transaction_id not in labelled]
    assert clean and not [r for r in clean if any(d.rule_id in core for d in r.discrepancies)]


def test_results_cover_every_transaction_sorted_and_deterministic(run) -> None:  # type: ignore[no-untyped-def]
    result, results, _ = run
    ids = [r.transaction_id for r in results]
    assert ids == sorted(ids) and len(ids) == 8000
    txs = [t for v in result.store.transactions_by_customer.values() for t in v]
    as_of = max(t.timestamp for t in txs)
    again = ReconciliationService().reconcile_many(txs, result.store.ledger_by_transaction, as_of)
    assert again == results


def test_reconcile_case_covers_focus_and_context(run) -> None:  # type: ignore[no-untyped-def]
    result, _, _ = run
    store = result.store
    cid = max(store.transactions_by_customer, key=lambda c: len(store.transactions_by_customer[c]))
    case = store.build_case(cid, context_days=60)
    results = ReconciliationService().reconcile_case(case)
    expected_ids = {
        c.transaction.transaction_id for c in [*case.focus_transactions, *case.context_transactions]
    }
    assert {r.transaction_id for r in results} == expected_ids


def test_source_records_unchanged_by_reconciliation(run) -> None:  # type: ignore[no-untyped-def]
    result, _, _ = run
    txs = [t for v in result.store.transactions_by_customer.values() for t in v][:500]
    snapshot = [t.model_dump() for t in txs]
    ledger_snapshot = {
        k: [x.model_dump() for x in v]
        for k, v in list(result.store.ledger_by_transaction.items())[:500]
    }
    ReconciliationService().reconcile_many(txs, result.store.ledger_by_transaction)
    assert [t.model_dump() for t in txs] == snapshot
    assert {
        k: [x.model_dump() for x in result.store.ledger_by_transaction[k]] for k in ledger_snapshot
    } == ledger_snapshot
