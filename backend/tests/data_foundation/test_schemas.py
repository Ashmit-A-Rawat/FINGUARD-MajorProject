from datetime import date, datetime

import pytest
from pydantic import ValidationError

from backend.app.schemas.domain import Customer, KYCRecord, Transaction


def _customer(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "customer_id": "CUST-000001",
        "name": "Ava Adams",
        "alternate_names": "A. Adams|Ava A",
        "date_of_birth": date(1990, 1, 1),
        "address": "1 Oak Street, Springfield, US",
        "country": "US",
        "occupation": "nurse",
        "account_type": "personal",
        "account_open_date": date(2020, 1, 1),
        "account_age_days": 100,
        "kyc_status": "verified",
    }
    return {**base, **overrides}


def test_alternate_names_split_from_pipe_string() -> None:
    assert Customer.model_validate(_customer()).alternate_names == ["A. Adams", "Ava A"]


def test_empty_alternate_names_become_empty_list() -> None:
    assert Customer.model_validate(_customer(alternate_names="")).alternate_names == []


def test_records_cannot_claim_to_be_real_data() -> None:
    with pytest.raises(ValidationError):
        Customer.model_validate(_customer(is_synthetic=False))


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Customer.model_validate(_customer(ssn="000-00-0000"))


def test_transaction_rejects_non_positive_amount() -> None:
    row = {
        "transaction_id": "T1",
        "customer_id": "C1",
        "timestamp": datetime(2025, 1, 1),
        "amount": 0,
        "currency": "USD",
        "transaction_type": "payment",
        "sender": "a",
        "receiver": "b",
        "sender_country": "US",
        "receiver_country": "US",
        "channel": "web",
    }
    with pytest.raises(ValidationError):
        Transaction.model_validate(row)


def test_kyc_expiry_must_follow_issue() -> None:
    row = {
        "document_id": "D1",
        "customer_id": "C1",
        "name": "A B",
        "date_of_birth": date(1990, 1, 1),
        "address": "x",
        "document_type": "passport",
        "document_number": "AB1234567",
        "issue_date": date(2024, 1, 1),
        "expiry_date": date(2023, 1, 1),
    }
    with pytest.raises(ValidationError):
        KYCRecord.model_validate(row)
