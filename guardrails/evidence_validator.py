"""Explicit evidence validation: does the cited evidence actually contain what the claim says?

No language model is involved. A claim is checked in two ways:
1. CITATIONS: every cited id must exist (case evidence or retrieved knowledge); a "fact" must
   cite case evidence.
2. CONTENT: every checkable token in the claim (identifiers, numbers, ISO dates, clock hours and
   minutes, currency codes; seconds are not checked) must appear in the cited evidence. Numbers
   match under rounding: "25" is supported by 25.31, "900.5" is not supported by 900.0.

Verdicts: ``supported`` (all checks pass), ``unsupported`` (a check failed), ``unverifiable``
(citations are fine but nothing in the claim is machine-checkable, or an inference's derived number
could not be found). Unverifiable is NOT the same as supported and is reported separately.

Limits: it verifies that quoted values exist in the cited evidence, not that the wording is a fair
reading of them (a claim can quote correct numbers and still draw a wrong conclusion). That is what
the policy floor and the human reviewer are for.
"""

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from backend.app.schemas.domain import Evidence
from guardrails.models import ClaimCheck, SupportStatus
from knowledge_base.models import RetrievedChunk
from llm.schemas import normalize_citation

CURRENCIES = {"USD", "EUR", "GBP", "INR", "AED", "SGD", "CAD", "AUD"}
_ID = re.compile(r"\b[A-Z][A-Z]+(?:-[A-Za-z0-9]+)+\b")
_KNOWLEDGE_ID = re.compile(r"\bKB-[A-Z]+-\d+(?::[a-z0-9\-]+)*(?::\d+)?\b")
_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}):(\d{2}))?")
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            "jan january",
            "feb february",
            "mar march",
            "apr april",
            "may",
            "jun june",
            "jul july",
            "aug august",
            "sep sept september",
            "oct october",
            "nov november",
            "dec december",
        ],
        start=1,
    )
    for m in names.split()
}
_MONTH_FORMS = sorted(
    {
        form
        for name in _MONTHS
        for form in (
            {name.capitalize(), name.upper()}
            if name == "may"
            else {name, name.capitalize(), name.upper()}
        )
    },
    key=len,
    reverse=True,
)
_MONTH_NAME = "|".join(_MONTH_FORMS)
_PROSE_DATE_MDY = re.compile(
    rf"\b({_MONTH_NAME})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b"
)
_PROSE_DATE_DMY = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_NAME})\.?(?:,?\s+(\d{{4}}))?\b"
)
_CLOCK_AMPM = re.compile(r"\b(\d{1,2}):(\d{2})\s?(am|pm|a\.m\.|p\.m\.)", re.I)
_AMPM = re.compile(r"\b(\d{1,2})\s?(am|pm|a\.m\.|p\.m\.)", re.I)
_NUMBER = re.compile(r"(?<![\w.:-])(\d[\d,]*(?:\.\d+)?)(?![\w:-])")


@dataclass(frozen=True)
class Tokens:
    ids: frozenset[str] = frozenset()
    numbers: tuple[tuple[float, int], ...] = ()  # (value, decimals as written)
    dates: frozenset[str] = frozenset()
    hours: frozenset[int] = frozenset()
    currencies: frozenset[str] = frozenset()
    times: frozenset[tuple[int, int]] = frozenset()  # (hour, minute); seconds are not checked
    month_days: frozenset[tuple[int, int]] = frozenset()  # (month, day) when no year is written

    def is_empty(self) -> bool:
        return not (
            self.ids
            or self.numbers
            or self.dates
            or self.hours
            or self.currencies
            or self.times
            or self.month_days
        )


def extract_tokens(text: str) -> Tokens:
    """Checkable content of a piece of text."""
    text = re.sub(r"\b[EK]:(?=[A-Za-z0-9])", "", text)  # citation prefixes are not content
    ids = set(_KNOWLEDGE_ID.findall(text))
    scrubbed = _KNOWLEDGE_ID.sub(" ", text)
    ids |= set(_ID.findall(scrubbed))
    scrubbed = _ID.sub(" ", scrubbed)
    dates: set[str] = set()
    month_days: set[tuple[int, int]] = set()
    hours: set[int] = set()
    times: set[tuple[int, int]] = set()
    for match in _DATE.finditer(scrubbed):
        dates.add(match.group(1))
        if match.group(2):
            hours.add(int(match.group(2)))
            times.add((int(match.group(2)), int(match.group(3))))
    scrubbed = _DATE.sub(" ", scrubbed)

    def prose_date(month_word: str, day: str, year: str | None) -> str:
        month = _MONTHS[month_word.lower()]
        if year:
            dates.add(f"{year}-{month:02d}-{int(day):02d}")
        else:
            month_days.add((month, int(day)))
        return " "

    scrubbed = _PROSE_DATE_MDY.sub(lambda m: prose_date(m[1], m[2], m[3]), scrubbed)
    scrubbed = _PROSE_DATE_DMY.sub(lambda m: prose_date(m[2], m[1], m[3]), scrubbed)
    for h, mnt, suffix in _CLOCK_AMPM.findall(scrubbed):
        hour = int(h) % 12 + (12 if suffix.lower().startswith("p") else 0)
        hours.add(hour)
        times.add((hour, int(mnt)))
    scrubbed = _CLOCK_AMPM.sub(" ", scrubbed)
    for h, m in _CLOCK.findall(scrubbed):
        hours.add(int(h) % 24)
        times.add((int(h) % 24, int(m)))
    scrubbed = _CLOCK.sub(" ", scrubbed)
    for h, suffix in _AMPM.findall(scrubbed):
        hour = int(h) % 12 + (12 if suffix.lower().startswith("p") else 0)
        hours.add(hour)
    scrubbed = _AMPM.sub(" ", scrubbed)
    numbers = []
    for raw in _NUMBER.findall(scrubbed):
        cleaned = raw.replace(",", "")
        numbers.append((float(cleaned), len(cleaned.split(".")[1]) if "." in cleaned else 0))
    currencies = {c for c in re.findall(r"\b[A-Z]{3}\b", scrubbed) if c in CURRENCIES}
    return Tokens(
        frozenset(ids),
        tuple(numbers),
        frozenset(dates),
        frozenset(hours),
        frozenset(currencies),
        frozenset(times),
        frozenset(month_days),
    )


def _keys(value: Any) -> Iterable[str]:
    """Field names, as words: 'transactions_in_prior_24_h' -> 'transactions in prior 24 h'."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield str(k).replace("_", " ")
            yield from _keys(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _keys(v)


def _walk(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk(v)
    else:
        yield value


@dataclass
class _Item:
    ref: str
    kind: str  # "evidence" | "knowledge"
    text: str
    tokens: Tokens = field(default_factory=Tokens)
    numbers: list[float] = field(default_factory=list)
    hours: set[int] = field(default_factory=set)


def _item_from_text(
    ref: str,
    kind: str,
    text: str,
    leaves: Iterable[Any] = (),
    payload: dict[str, Any] | None = None,
) -> _Item:
    """Everything an evidence item or chunk contains that a claim could legitimately quote."""
    tokens = extract_tokens(text)
    numbers = [n for n, _ in tokens.numbers]
    ids = set(tokens.ids) | {ref}
    dates = set(tokens.dates)
    hours = set(tokens.hours)
    times = set(tokens.times)
    month_days = set(tokens.month_days)
    currencies = set(tokens.currencies)
    if payload and isinstance(payload.get("hour_of_day"), int):
        hours.add(int(payload["hour_of_day"]))
    for leaf in leaves:
        if isinstance(leaf, bool):
            continue
        if isinstance(leaf, (int, float)):
            numbers.append(float(leaf))
        elif isinstance(leaf, str):
            sub = extract_tokens(leaf)
            numbers += [n for n, _ in sub.numbers]
            ids |= set(sub.ids)
            dates |= set(sub.dates)
            hours |= set(sub.hours)
            times |= set(sub.times)
            month_days |= set(sub.month_days)
            currencies |= set(sub.currencies)
            for stamp in re.findall(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", leaf):
                moment = datetime.fromisoformat(stamp.replace(" ", "T"))
                hours.add(moment.hour)
                times.add((moment.hour, moment.minute))
    merged = Tokens(
        frozenset(ids),
        tuple((n, 0) for n in numbers),
        frozenset(dates),
        frozenset(hours),
        frozenset(currencies),
        frozenset(times),
        frozenset(month_days),
    )
    return _Item(ref, kind, text, merged, numbers, hours)


class EvidenceIndex:
    """Lookup of what each evidence item and retrieved chunk actually contains."""

    def __init__(self, evidence: Sequence[Evidence], chunks: Sequence[RetrievedChunk] = ()) -> None:
        self.items: dict[str, _Item] = {}
        for e in evidence:
            leaves = [*_walk(e.payload), *_keys(e.payload)]
            payload_text = json.dumps(e.payload, default=str)
            self.items[e.evidence_id] = _item_from_text(
                e.evidence_id,
                "evidence",
                f"{e.description} data: {payload_text}",
                leaves,
                e.payload,
            )
        for c in chunks:
            self.items[c.chunk_id] = _item_from_text(c.chunk_id, "knowledge", c.text)

    def resolve(self, reference: str) -> _Item | None:
        return self.items.get(normalize_citation(reference))

    def case_items(self) -> list[_Item]:
        return [i for i in self.items.values() if i.kind == "evidence"]


def _numbers_supported(tokens: Tokens, items: Sequence[_Item]) -> list[float]:
    """Numbers in the claim that no item contains (rounding-consistent match)."""
    available = [n for i in items for n in i.numbers]
    missing = []
    for value, decimals in tokens.numbers:
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9
        if not any(abs(value - a) <= tolerance for a in available):
            missing.append(value)
    return missing


@dataclass
class _Missing:
    ids: list[str]
    numbers: list[float]
    dates: list[str]
    hours: list[int]
    currencies: list[str]
    times: list[tuple[int, int]]
    month_days: list[tuple[int, int]]

    def any(self) -> bool:
        return bool(
            self.ids
            or self.numbers
            or self.dates
            or self.hours
            or self.currencies
            or self.times
            or self.month_days
        )

    def describe(self) -> str:
        parts = []
        if self.ids:
            parts.append("identifiers " + ", ".join(sorted(self.ids)))
        if self.numbers:
            parts.append("numbers " + ", ".join(f"{n:g}" for n in self.numbers))
        if self.dates:
            parts.append("dates " + ", ".join(sorted(self.dates)))
        if self.hours:
            parts.append("hours " + ", ".join(str(h) for h in sorted(self.hours)))
        if self.times:
            parts.append("times " + ", ".join(f"{h:02d}:{m:02d}" for h, m in sorted(self.times)))
        if self.month_days:
            parts.append(
                "dates " + ", ".join(f"{m:02d}-{d:02d}" for m, d in sorted(self.month_days))
            )
        if self.currencies:
            parts.append("currencies " + ", ".join(sorted(self.currencies)))
        return "; ".join(parts)


def _missing(tokens: Tokens, items: Sequence[_Item]) -> _Missing:
    have_ids = {i for it in items for i in it.tokens.ids}
    have_dates = {d for it in items for d in it.tokens.dates}
    have_hours = {h for it in items for h in it.hours | set(it.tokens.hours)}
    have_currencies = {c for it in items for c in it.tokens.currencies}
    have_times = {t for it in items for t in it.tokens.times}
    have_month_days = {(int(d[5:7]), int(d[8:10])) for it in items for d in it.tokens.dates} | {
        md for it in items for md in it.tokens.month_days
    }
    return _Missing(
        ids=sorted(tokens.ids - have_ids),
        numbers=_numbers_supported(tokens, items),
        dates=sorted(tokens.dates - have_dates),
        hours=sorted(tokens.hours - have_hours),
        currencies=sorted(tokens.currencies - have_currencies),
        times=sorted(tokens.times - have_times),
        month_days=sorted(tokens.month_days - have_month_days),
    )


def check_claim(
    index_in_report: int, claim: str, kind: str, evidence_ids: Sequence[str], index: EvidenceIndex
) -> ClaimCheck:
    refs = list(evidence_ids)

    def result(status: SupportStatus, reason: str) -> ClaimCheck:
        return ClaimCheck(
            index=index_in_report,
            claim=claim,
            kind=kind,
            evidence_ids=refs,
            status=status,
            reason=reason,
        )

    resolved = [(r, index.resolve(r)) for r in refs]
    unknown = [r for r, item in resolved if item is None]
    if unknown:
        return result("unsupported", f"cites evidence that does not exist: {', '.join(unknown)}")
    items = [item for _, item in resolved if item is not None]
    case_cited = [i for i in items if i.kind == "evidence"]
    if kind == "fact" and not case_cited:
        return result("unsupported", "a fact must cite at least one item of case evidence")
    if not items:
        return result("unsupported", "no evidence is cited")

    tokens = extract_tokens(claim)
    if tokens.is_empty():
        return result(
            "unverifiable", "citations exist, but the claim has no numbers or identifiers to check"
        )
    missing = _missing(tokens, items)
    if not missing.any():
        return result(
            "supported",
            "all identifiers, numbers, dates and currencies appear in the cited evidence",
        )

    elsewhere = [
        it.ref for it in index.case_items() if it not in items and not _missing(tokens, [it]).any()
    ]
    hint = (
        f" (they do appear in {', '.join(elsewhere[:2])}, which the claim does not cite)"
        if elsewhere
        else ""
    )
    reason = f"not found in the cited evidence: {missing.describe()}{hint}"
    derived_only = not (missing.ids or missing.dates or missing.currencies)
    if kind == "inference" and derived_only:
        return result(
            "unverifiable",
            f"derived value(s) could not be matched to the cited evidence: {missing.describe()}",
        )
    return result("unsupported", reason)


def check_text_grounded(text: str, index: EvidenceIndex) -> tuple[bool, str]:
    """Every checkable token in ``text`` exists somewhere in the case evidence."""
    tokens = extract_tokens(text)
    if tokens.is_empty():
        return True, "no checkable content"
    missing = _missing(tokens, index.case_items())
    return (
        not missing.any(),
        "grounded" if not missing.any() else f"not in any case evidence: {missing.describe()}",
    )
