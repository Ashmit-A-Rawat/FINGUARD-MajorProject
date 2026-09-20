import re

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")


def normalize_document_number(raw: str) -> str:
    cleaned = _NON_ALNUM.sub("", raw).upper()
    if not cleaned:
        raise ValueError(f"document number has no alphanumeric characters: {raw!r}")
    return cleaned
