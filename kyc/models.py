"""KYC engine data contracts and configuration."""

from dataclasses import dataclass
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from data_pipeline.consolidation.models import CanonicalKYCDocument
from data_pipeline.normalization.models import NormalizedAddress, NormalizedName

Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class QueryIdentity:
    """The identity attributes being verified against the customer master."""

    name: NormalizedName
    date_of_birth: date | None
    address: NormalizedAddress | None

    @classmethod
    def from_document(cls, document: CanonicalKYCDocument) -> "QueryIdentity":
        return cls(document.name, document.record.date_of_birth, document.address)


class MatcherConfig(BaseModel):
    top_k: int = Field(20, ge=1)  # candidates kept after first-stage retrieval
    hybrid_alpha: float = Field(0.5, ge=0, le=1)  # weight of BM25 in the BM25/dense fusion
    weight_name: float = Field(0.5, ge=0)
    weight_structured: float = Field(0.5, ge=0)
    dob_weight: float = Field(0.6, ge=0)
    address_weight: float = Field(0.4, ge=0)
    match_threshold: float = Field(0.65, ge=0, le=1)  # decision threshold on final_score
    high_confidence_margin: float = Field(0.10, ge=0)


class KYCMatchResult(BaseModel):
    """Explainable per-candidate output: every intermediate score is exposed."""

    candidate_id: str
    lexical_score: float  # normalised BM25 over name variants
    semantic_score: float  # cosine similarity of name embeddings
    reranker_score: float | None  # learned name reranker; None when not used
    structured_match_score: float | None  # DOB / address verification; None when not used
    final_score: float
    is_match: bool
    confidence: Confidence
    match_reasons: list[str]
    contradictory_evidence: list[str]
    signals: dict[str, float]  # all raw first-stage signals (bm25, bm25_char, fuzzy, dense, hybrid)
