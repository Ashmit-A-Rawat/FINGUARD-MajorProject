"""Record-level cleaning. Every change is counted so nothing happens silently."""

import unicodedata
from collections import Counter
from collections.abc import Mapping

# Whole-field values treated as missing. "NA" is deliberately absent: it is a valid country code.
NULL_TOKENS = frozenset({"", "nan", "null", "none", "n/a"})


def clean_value(value: str | None, actions: Counter[str]) -> str | None:
    if value is None:
        return None
    text = unicodedata.normalize("NFC", value)
    if text != value:
        actions["unicode_nfc"] += 1
    stripped = text.strip()
    if stripped != text:
        actions["trim_whitespace"] += 1
    collapsed = " ".join(stripped.split())
    if collapsed != stripped:
        actions["collapse_whitespace"] += 1
    if collapsed.casefold() in NULL_TOKENS:
        actions["null_token_to_none"] += 1
        return None
    return collapsed


def clean_record(record: Mapping[str, str | None], actions: Counter[str]) -> dict[str, str | None]:
    return {key: clean_value(value, actions) for key, value in record.items()}
