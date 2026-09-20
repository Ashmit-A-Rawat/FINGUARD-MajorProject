from datetime import date

from backend.app.schemas.domain import Customer
from data_pipeline.consolidation.canonicalize import canonicalize_customer
from data_pipeline.normalization.addresses import normalize_address
from data_pipeline.normalization.names import normalize_name
from kyc.models import MatcherConfig, QueryIdentity
from kyc.scoring.structured import verify_structured


def _customer(dob: date = date(1985, 3, 9), address: str = "12 Oak Street, Springfield, US"):  # type: ignore[no-untyped-def]
    return canonicalize_customer(
        Customer(
            customer_id="C1",
            name="Ava Adams",
            date_of_birth=dob,
            address=address,
            country="US",
            occupation="nurse",
            account_type="personal",
            account_open_date=date(2020, 1, 1),
            account_age_days=100,
            kyc_status="verified",
        )
    )


def _query(dob: date | None, address: str | None) -> QueryIdentity:
    return QueryIdentity(
        normalize_name("Ava Adams"), dob, normalize_address(address) if address else None
    )


CFG = MatcherConfig()


def test_perfect_agreement() -> None:
    v = verify_structured(_query(date(1985, 3, 9), "12 Oak St., Springfield, us"), _customer(), CFG)
    assert v.score == 1.0 and not v.contradictions
    assert {"dob_exact", "address_exact"} <= set(v.reasons)


def test_dob_conflict_is_reported_as_contradiction() -> None:
    v = verify_structured(
        _query(date(1990, 1, 1), "12 Oak Street, Springfield, US"), _customer(), CFG
    )
    assert v.dob_score == 0.0 and v.address_score == 1.0
    assert v.score == 0.4  # 0.6*0 + 0.4*1
    assert any(c.startswith("dob_mismatch") for c in v.contradictions)


def test_day_month_transposition_gets_partial_credit_and_is_flagged() -> None:
    v = verify_structured(_query(date(1985, 9, 3), None), _customer(), CFG)
    assert v.dob_score == 0.5 and "dob_day_month_transposed" in v.contradictions


def test_address_components_are_scored_individually() -> None:
    v = verify_structured(
        _query(date(1985, 3, 9), "99 Oak Street, Springfield, US"), _customer(), CFG
    )
    assert v.address_score == 0.75  # everything but the house number
    assert any("address_number_mismatch" in c for c in v.contradictions)
    assert any("address_partial" in r for r in v.reasons)


def test_different_country_is_flagged() -> None:
    v = verify_structured(
        _query(date(1985, 3, 9), "12 Oak Street, Springfield, GB"), _customer(), CFG
    )
    assert any("address_country_mismatch" in c for c in v.contradictions)


def test_missing_query_fields_are_neutral_not_penalised() -> None:
    v = verify_structured(_query(None, None), _customer(), CFG)
    assert v.score == 0.0 and v.dob_score is None and v.address_score is None
    only_dob = verify_structured(_query(date(1985, 3, 9), None), _customer(), CFG)
    assert only_dob.score == 1.0  # weights renormalised over available evidence
