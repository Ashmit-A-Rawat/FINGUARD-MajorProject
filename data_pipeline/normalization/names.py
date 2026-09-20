"""Canonical person-name form used for matching. The raw name is always preserved alongside."""

import re

from data_pipeline.normalization.models import NormalizedName
from data_pipeline.normalization.text import strip_accents

_PUNCTUATION = re.compile(r"[^\w\s'-]", re.UNICODE)


def normalize_name(raw: str) -> NormalizedName:
    """Casefold, strip accents/punctuation, and convert 'Last, First Middle' to 'first middle last'.

    Only a comma triggers reordering; otherwise token order is kept. Raises ValueError if
    nothing usable remains, so callers must handle (quarantine) such records explicitly.
    """
    text = strip_accents(raw).casefold()
    reordered = False
    if "," in text:
        surname, _, rest = text.partition(",")
        if surname.strip() and rest.strip():
            text = f"{rest} {surname}"
            reordered = True
    text = _PUNCTUATION.sub(" ", text.replace(".", " "))
    tokens = text.split()
    if not tokens:
        raise ValueError(f"name has no usable tokens: {raw!r}")
    return NormalizedName(
        raw=raw,
        canonical=" ".join(tokens),
        tokens=tokens,
        first=tokens[0],
        middle=tokens[1:-1],
        last=tokens[-1],
        has_initials=any(len(t) == 1 for t in tokens),
        reordered=reordered,
    )
