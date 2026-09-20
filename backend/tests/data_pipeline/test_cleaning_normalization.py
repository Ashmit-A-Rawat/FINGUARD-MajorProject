from collections import Counter

import pytest

from data_pipeline.cleaning.cleaner import clean_record, clean_value
from data_pipeline.normalization.addresses import normalize_address
from data_pipeline.normalization.identifiers import normalize_document_number
from data_pipeline.normalization.names import normalize_name


def test_clean_value_counts_every_change() -> None:
    actions: Counter[str] = Counter()
    assert clean_value("  Ava \t  Adams ", actions) == "Ava Adams"
    assert actions == {"trim_whitespace": 1, "collapse_whitespace": 1}


def test_clean_value_unicode_composition() -> None:
    actions: Counter[str] = Counter()
    assert clean_value("José", actions) == "José"
    assert actions["unicode_nfc"] == 1


@pytest.mark.parametrize("token", ["", "  ", "NaN", "NULL", "None", "n/a"])
def test_null_tokens_become_none(token: str) -> None:
    assert clean_value(token, Counter()) is None


def test_na_country_code_is_not_treated_as_missing() -> None:
    assert clean_value("NA", Counter()) == "NA"


def test_clean_record_leaves_clean_values_untouched() -> None:
    actions: Counter[str] = Counter()
    record = {"a": "x", "b": None}
    assert clean_record(record, actions) == record
    assert not actions


def test_name_basic_and_raw_preserved() -> None:
    n = normalize_name("  Ava   ADAMS ")
    assert n.canonical == "ava adams"
    assert n.raw == "  Ava   ADAMS "
    assert (n.first, n.middle, n.last) == ("ava", [], "adams")


def test_name_last_comma_first_is_reordered() -> None:
    n = normalize_name("Chaudhary, Rahul Kenji")
    assert n.canonical == "rahul kenji chaudhary"
    assert n.reordered and n.middle == ["kenji"]


def test_name_initials_and_accents() -> None:
    n = normalize_name("J. Élodie Müller")
    assert n.canonical == "j elodie muller"
    assert n.has_initials


def test_name_keeps_hyphen_and_apostrophe() -> None:
    assert normalize_name("Anne-Marie O'Neil").canonical == "anne-marie o'neil"


def test_name_without_usable_tokens_raises() -> None:
    with pytest.raises(ValueError):
        normalize_name(" ., ")


def test_address_structured_and_suffix_standardised() -> None:
    a = normalize_address("537  Maple Avenue, Kampong   Baru, sg")
    assert a.parsed and a.canonical == "537 maple ave, kampong baru, SG"
    assert normalize_address("537 Maple Ave., Kampong Baru, SG").canonical == a.canonical


def test_address_unparseable_is_kept_and_flagged() -> None:
    a = normalize_address("Flat 4B, The Old Mill")
    assert not a.parsed and a.raw == "Flat 4B, The Old Mill"


def test_document_number_normalisation() -> None:
    assert normalize_document_number("ab-123 4567") == "AB1234567"
    with pytest.raises(ValueError):
        normalize_document_number("--")
