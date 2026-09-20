"""Address normalization. Unparseable addresses are kept (flagged), never guessed at."""

import re

from data_pipeline.normalization.models import NormalizedAddress
from data_pipeline.normalization.text import strip_accents

STREET_SUFFIXES = {
    "street": "st",
    "st": "st",
    "road": "rd",
    "rd": "rd",
    "avenue": "ave",
    "ave": "ave",
    "lane": "ln",
    "ln": "ln",
    "drive": "dr",
    "dr": "dr",
}
_STRUCTURED = re.compile(
    r"^(?P<number>\d+)\s+(?P<street>[^,]+),\s*(?P<city>[^,]+),\s*(?P<country>[A-Za-z]{2})$"
)


def normalize_address(raw: str) -> NormalizedAddress:
    text = " ".join(strip_accents(raw).split())
    match = _STRUCTURED.match(text)
    if match is None:
        return NormalizedAddress(raw=raw, canonical=text.casefold().replace(".", ""), parsed=False)
    street_tokens = match["street"].casefold().replace(".", "").split()
    street_tokens[-1] = STREET_SUFFIXES.get(street_tokens[-1], street_tokens[-1])
    street = " ".join(street_tokens)
    city = " ".join(match["city"].casefold().replace(".", "").split())
    country = match["country"].upper()
    return NormalizedAddress(
        raw=raw,
        canonical=f"{match['number']} {street}, {city}, {country}",
        parsed=True,
        number=match["number"],
        street=street,
        city=city,
        country=country,
    )
