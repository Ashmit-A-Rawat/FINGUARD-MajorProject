"""Stage G: structured attribute verification (date of birth, address) with explicit evidence."""

from dataclasses import dataclass, field
from datetime import date

from rapidfuzz import fuzz

from data_pipeline.consolidation.models import CanonicalCustomer
from data_pipeline.normalization.models import NormalizedAddress
from kyc.models import MatcherConfig, QueryIdentity

_ADDRESS_COMPONENT_WEIGHTS = {"number": 0.25, "street": 0.25, "city": 0.30, "country": 0.20}


@dataclass
class StructuredVerification:
    score: float  # weighted DOB/address agreement in [0, 1]
    dob_score: float | None
    address_score: float | None
    reasons: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)


def _compare_dob(query: date, candidate: date) -> tuple[float, str, bool]:
    """Return (score, evidence, is_contradiction)."""
    if query == candidate:
        return 1.0, "dob_exact", False
    if query.year == candidate.year and (query.day, query.month) == (
        candidate.month,
        candidate.day,
    ):
        return 0.5, "dob_day_month_transposed", True
    return 0.0, f"dob_mismatch: {query.isoformat()} vs {candidate.isoformat()}", True


def _compare_address(
    query: NormalizedAddress, candidate: NormalizedAddress
) -> tuple[float, list[str], list[str]]:
    if query.canonical == candidate.canonical:
        return 1.0, ["address_exact"], []
    if not (query.parsed and candidate.parsed):
        ratio = fuzz.ratio(query.canonical, candidate.canonical) / 100.0
        return ratio, [], [f"address_unstructured_similarity={ratio:.2f}"]
    matched: list[str] = []
    contradictions: list[str] = []
    score = 0.0
    for component, weight in _ADDRESS_COMPONENT_WEIGHTS.items():
        a, b = getattr(query, component), getattr(candidate, component)
        if a == b:
            score += weight
            matched.append(component)
        else:
            contradictions.append(f"address_{component}_mismatch: {a!r} vs {b!r}")
    reasons = [f"address_partial: matching {', '.join(matched)}"] if matched else []
    return score, reasons, contradictions


def verify_structured(
    query: QueryIdentity, candidate: CanonicalCustomer, config: MatcherConfig
) -> StructuredVerification:
    parts: list[tuple[float, float]] = []  # (weight, score)
    reasons: list[str] = []
    contradictions: list[str] = []
    dob_score: float | None = None
    address_score: float | None = None

    if query.date_of_birth is not None:
        dob_score, evidence, contradicts = _compare_dob(
            query.date_of_birth, candidate.customer.date_of_birth
        )
        (contradictions if contradicts else reasons).append(evidence)
        parts.append((config.dob_weight, dob_score))
    else:
        reasons.append("dob_missing_in_query (neutral)")

    if query.address is not None:
        address_score, addr_reasons, addr_contradictions = _compare_address(
            query.address, candidate.address
        )
        reasons += addr_reasons
        contradictions += addr_contradictions
        parts.append((config.address_weight, address_score))
    else:
        reasons.append("address_missing_in_query (neutral)")

    total_weight = sum(w for w, _ in parts)
    score = sum(w * s for w, s in parts) / total_weight if total_weight > 0 else 0.0
    return StructuredVerification(score, dob_score, address_score, reasons, contradictions)
