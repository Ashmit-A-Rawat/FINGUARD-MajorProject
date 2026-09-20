from datetime import datetime, timedelta

import pytest

from backend.app.schemas.domain import Discrepancy, LedgerRecord, Severity, Transaction
from reconciliation.config import ReconciliationConfig
from reconciliation.service import ReconciliationService

T0 = datetime(2025, 3, 1, 12, 0, 0)
SERVICE = ReconciliationService()


def tx(**overrides: object) -> Transaction:
    base: dict[str, object] = {
        "transaction_id": "TXN-1",
        "customer_id": "C1",
        "timestamp": T0,
        "amount": 100.0,
        "currency": "USD",
        "transaction_type": "payment",
        "sender": "ACC-1",
        "receiver": "MRC-1",
        "sender_country": "US",
        "receiver_country": "US",
        "channel": "web",
        "reference_id": "REF-1",
    }
    return Transaction.model_validate({**base, **overrides})


def led(**overrides: object) -> LedgerRecord:
    base: dict[str, object] = {
        "ledger_id": "LED-1",
        "transaction_id": "TXN-1",
        "posted_amount": 100.0,
        "currency": "USD",
        "posting_timestamp": T0 + timedelta(minutes=30),
        "settlement_status": "settled",
        "reference_id": "REF-1",
    }
    return LedgerRecord.model_validate({**base, **overrides})


def rules_of(result_discrepancies: list[Discrepancy]) -> list[str]:
    return [d.rule_id for d in result_discrepancies]


def test_clean_pair_is_reconciled() -> None:
    r = SERVICE.reconcile(tx(), [led()])
    assert r.status == "reconciled" and r.discrepancies == [] and r.ledger_id == "LED-1"


def test_missing_ledger_entry_only_reports_that_rule() -> None:
    r = SERVICE.reconcile(tx(), [])
    assert r.status == "discrepancy" and r.ledger_id is None
    (d,) = r.discrepancies
    assert (d.rule_id, d.field, d.severity) == ("REC-001", "ledger_record", Severity.HIGH)
    assert d.expected == "1 ledger entry" and d.actual == "none"


def test_ledger_records_of_other_transactions_are_ignored() -> None:
    r = SERVICE.reconcile(tx(), [led(transaction_id="TXN-2")])
    assert rules_of(r.discrepancies) == ["REC-001"]


@pytest.mark.parametrize(
    ("posted", "expected_rules"),
    [(100.00, []), (100.01, []), (99.99, []), (100.02, ["REC-003"]), (99.98, ["REC-003"])],
)
def test_amount_tolerance_boundary_is_one_cent(posted: float, expected_rules: list[str]) -> None:
    assert (
        rules_of(SERVICE.reconcile(tx(), [led(posted_amount=posted)]).discrepancies)
        == expected_rules
    )


def test_amount_comparison_is_immune_to_float_noise() -> None:
    assert SERVICE.reconcile(tx(amount=0.1 + 0.2), [led(posted_amount=0.3)]).status == "reconciled"


@pytest.mark.parametrize(
    ("posted", "severity"),
    [
        (100.5, Severity.LOW),
        (101.5, Severity.MEDIUM),
        (110.0, Severity.HIGH),
        (50.0, Severity.HIGH),
    ],
)
def test_amount_severity_scales_with_relative_difference(posted: float, severity: Severity) -> None:
    (d,) = SERVICE.reconcile(tx(), [led(posted_amount=posted)]).discrepancies
    assert d.rule_id == "REC-003" and d.severity == severity


def test_amount_discrepancy_carries_full_evidence() -> None:
    (d,) = SERVICE.reconcile(tx(), [led(posted_amount=112.5)]).discrepancies
    assert (d.field, d.expected, d.actual) == ("amount", "100.00", "112.50")
    assert d.difference == "+12.50 (+12.5%)" and "112.50" in d.explanation


def test_currency_and_amount_mismatch_are_both_reported_in_rule_order() -> None:
    r = SERVICE.reconcile(tx(), [led(currency="EUR", posted_amount=150.0)])
    assert rules_of(r.discrepancies) == ["REC-003", "REC-004"]
    assert r.discrepancies[1].expected == "USD" and r.discrepancies[1].actual == "EUR"


@pytest.mark.parametrize(
    ("tx_ref", "led_ref", "expected_rules"),
    [
        ("REF-1", "REF-1", []),
        ("REF-1", None, ["REC-005"]),
        (None, "REF-1", ["REC-005"]),
        (None, None, []),
        ("REF-1", "REF-9", ["REC-006"]),
    ],
)
def test_reference_rules(
    tx_ref: str | None, led_ref: str | None, expected_rules: list[str]
) -> None:
    r = SERVICE.reconcile(tx(reference_id=tx_ref), [led(reference_id=led_ref)])
    assert rules_of(r.discrepancies) == expected_rules


def test_duplicate_posting_uses_earliest_as_primary_and_lists_all_ids() -> None:
    later = led(ledger_id="LED-2", posting_timestamp=T0 + timedelta(minutes=31))
    r = SERVICE.reconcile(tx(), [later, led()])  # given out of order on purpose
    assert r.ledger_id == "LED-1"
    (d,) = r.discrepancies
    assert d.rule_id == "REC-002" and d.actual == "2" and d.difference == "+1"
    assert "LED-1" in d.explanation and "LED-2" in d.explanation


def test_other_rules_apply_to_the_primary_record_only() -> None:
    bad_duplicate = led(
        ledger_id="LED-2", posted_amount=1.0, posting_timestamp=T0 + timedelta(hours=1)
    )
    assert rules_of(SERVICE.reconcile(tx(), [led(), bad_duplicate]).discrepancies) == ["REC-002"]


def test_late_posting_boundary_and_severity() -> None:
    limit = timedelta(days=3)
    assert SERVICE.reconcile(tx(), [led(posting_timestamp=T0 + limit)]).status == "reconciled"
    (medium,) = SERVICE.reconcile(
        tx(), [led(posting_timestamp=T0 + limit + timedelta(seconds=1))]
    ).discrepancies
    assert (medium.rule_id, medium.severity) == ("REC-008", Severity.MEDIUM)
    (high,) = SERVICE.reconcile(
        tx(), [led(posting_timestamp=T0 + timedelta(days=10))]
    ).discrepancies
    assert high.severity == Severity.HIGH


def test_posting_before_transaction_is_flagged() -> None:
    (d,) = SERVICE.reconcile(tx(), [led(posting_timestamp=T0 - timedelta(seconds=1))]).discrepancies
    assert (d.rule_id, d.severity, d.field) == ("REC-007", Severity.HIGH, "posting_timestamp")


def test_settlement_status_rules() -> None:
    assert rules_of(SERVICE.reconcile(tx(), [led(settlement_status="failed")]).discrepancies) == [
        "REC-009"
    ]
    fresh = SERVICE.reconcile(
        tx(), [led(settlement_status="pending")], as_of=T0 + timedelta(days=1)
    )
    assert fresh.status == "reconciled"
    stale = SERVICE.reconcile(
        tx(), [led(settlement_status="pending")], as_of=T0 + timedelta(days=4)
    )
    assert rules_of(stale.discrepancies) == ["REC-010"]
    no_clock = SERVICE.reconcile(tx(), [led(settlement_status="pending")])
    assert no_clock.status == "reconciled"  # cannot judge staleness without as_of


def test_status_checks_can_be_disabled() -> None:
    service = ReconciliationService(ReconciliationConfig(check_settlement_status=False))
    assert service.reconcile(tx(), [led(settlement_status="failed")]).status == "reconciled"


def test_custom_tolerances_are_respected() -> None:
    service = ReconciliationService(ReconciliationConfig(amount_tolerance_cents=500))
    assert service.reconcile(tx(), [led(posted_amount=104.99)]).status == "reconciled"
    assert service.reconcile(tx(), [led(posted_amount=105.01)]).status == "discrepancy"


def test_inputs_are_never_modified_and_results_are_deterministic() -> None:
    t, ledger = tx(), [led(posted_amount=150.0, reference_id=None), led(ledger_id="LED-2")]
    before_t, before_l = t.model_copy(deep=True), [x.model_copy(deep=True) for x in ledger]
    first = SERVICE.reconcile(t, ledger)
    second = SERVICE.reconcile(t, list(reversed(ledger)))
    assert t == before_t and ledger == before_l
    assert first == second


def test_config_is_frozen() -> None:
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        ReconciliationConfig().amount_tolerance_cents = 99  # type: ignore[misc]
