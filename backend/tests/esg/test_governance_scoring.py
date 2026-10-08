"""Governance-risk evidence: deterministic checks and its wiring into the engine floor.

Added in response to review feedback asking us to explore an ESG score as an additional
parameter. Scope is the Governance pillar only; see esg/governance_scoring.py for why.
"""

from datetime import datetime, timedelta

from backend.app.schemas.domain import Channel, Transaction, TransactionType
from esg.governance_scoring import GovernanceConfig, governance_evidence, score_governance
from guardrails.policy_checks import PolicyConfig, engine_floor

START = datetime(2025, 1, 1, 9, 0, 0)


def tx(
    tid: str,
    *,
    hours: float = 0,
    amount: float = 500.0,
    currency: str = "USD",
    receiver_country: str = "US",
    sender_country: str = "US",
    ttype: TransactionType = TransactionType.PAYMENT,
) -> Transaction:
    return Transaction(
        transaction_id=tid,
        customer_id="CUST-1",
        timestamp=START + timedelta(hours=hours),
        amount=amount,
        currency=currency,
        transaction_type=ttype,
        sender="ACC-1",
        receiver="MRC-1",
        sender_country=sender_country,
        receiver_country=receiver_country,
        channel=Channel.WEB,
        is_synthetic=True,
    )


def test_no_flags_on_an_ordinary_transaction() -> None:
    history = [tx(f"H{i}", hours=-i) for i in range(1, 6)]
    facts = score_governance(tx("T1"), history)
    assert not facts["new_cross_border_counterparty"]
    assert not facts["structuring_suspected"]
    assert not facts["flagged"]


def test_new_cross_border_counterparty_is_flagged() -> None:
    history = [tx(f"H{i}", hours=-i, receiver_country="US") for i in range(1, 4)]
    focus = tx("T1", receiver_country="PA", sender_country="US")  # never seen before, cross-border
    facts = score_governance(focus, history)
    assert facts["new_cross_border_counterparty"]
    assert facts["counterparty_country"] == "PA"


def test_a_country_already_seen_is_not_flagged() -> None:
    history = [
        tx(f"H{i}", hours=-i, receiver_country="PA", sender_country="US") for i in range(1, 4)
    ]
    focus = tx("T1", receiver_country="PA", sender_country="US")
    assert not score_governance(focus, history)["new_cross_border_counterparty"]


def test_domestic_transaction_to_a_new_counterparty_is_not_flagged() -> None:
    """A new counterparty is only a governance signal when it is ALSO cross-border."""
    history = [
        tx(f"H{i}", hours=-i, receiver_country="US", sender_country="US") for i in range(1, 4)
    ]
    focus = tx("T1", receiver_country="US", sender_country="US")
    assert not score_governance(focus, history)["new_cross_border_counterparty"]


def test_structuring_pattern_is_flagged() -> None:
    history = [tx(f"H{i}", hours=-i, amount=9500.0) for i in range(1, 3)]  # 2 prior, just under 10k
    focus = tx("T1", amount=9800.0)  # the 3rd -> hits the default min_count of 3
    facts = score_governance(focus, history)
    assert facts["structuring_suspected"]
    assert facts["sub_threshold_transactions_in_window"] == 3


def test_structuring_needs_the_minimum_count() -> None:
    history = [tx("H1", hours=-1, amount=9500.0)]  # only 1 prior + focus = 2, below min_count=3
    facts = score_governance(tx("T1", amount=9800.0), history)
    assert not facts["structuring_suspected"]


def test_structuring_window_excludes_old_transactions() -> None:
    old = [tx(f"H{i}", hours=-24 * 90 - i, amount=9500.0) for i in range(1, 3)]  # 90 days ago
    facts = score_governance(
        tx("T1", amount=9800.0), old, GovernanceConfig(structuring_window_days=30)
    )
    assert not facts["structuring_suspected"]


def test_structuring_ignores_a_different_currency() -> None:
    history = [tx(f"H{i}", hours=-i, amount=9500.0, currency="EUR") for i in range(1, 3)]
    facts = score_governance(tx("T1", amount=9800.0, currency="USD"), history)
    assert not facts["structuring_suspected"]


def test_amount_at_or_above_the_threshold_does_not_count() -> None:
    history = [tx(f"H{i}", hours=-i, amount=10_000.0) for i in range(1, 3)]
    facts = score_governance(tx("T1", amount=10_000.0), history)
    assert not facts["structuring_suspected"]


def test_governance_evidence_shape_and_id() -> None:
    history = [tx(f"H{i}", hours=-i, receiver_country="US") for i in range(1, 4)]
    ev = governance_evidence(tx("T1", receiver_country="PA"), history, START)
    assert ev.evidence_id == "ESG-T1"
    assert ev.source.value == "esg"
    assert "PA" in ev.description


def test_governance_evidence_unflagged_case_has_a_neutral_description() -> None:
    history = [tx(f"H{i}", hours=-i) for i in range(1, 4)]
    ev = governance_evidence(tx("T1"), history, START)
    assert "none triggered" in ev.description


def test_engine_floor_reacts_to_a_flagged_governance_item() -> None:
    history = [tx(f"H{i}", hours=-i, receiver_country="US") for i in range(1, 4)]
    ev = governance_evidence(tx("T1", receiver_country="PA"), history, START)
    floor, reasons = engine_floor([ev], PolicyConfig())
    assert floor is not None
    assert any("governance" in r for r in reasons)


def test_engine_floor_ignores_governance_when_switched_off() -> None:
    history = [tx(f"H{i}", hours=-i, receiver_country="US") for i in range(1, 4)]
    ev = governance_evidence(tx("T1", receiver_country="PA"), history, START)
    floor, _ = engine_floor([ev], PolicyConfig(esg_governance_forces_review=False))
    assert floor is None


def test_engine_floor_stays_clear_on_an_unflagged_governance_item() -> None:
    history = [tx(f"H{i}", hours=-i) for i in range(1, 4)]
    ev = governance_evidence(tx("T1"), history, START)
    floor, _ = engine_floor([ev], PolicyConfig())
    assert floor is None
