"""KYC entity matcher: verifies a KYC document against the customer master, with evidence."""

from collections.abc import Sequence

from data_pipeline.consolidation.models import CanonicalCustomer, CanonicalKYCDocument
from kyc.models import Confidence, KYCMatchResult, MatcherConfig, QueryIdentity
from kyc.reranking.reranker import LearnedReranker
from kyc.semantic.embedder import Embedder
from kyc.signals import SignalStore
from kyc.systems import CandidateScorer, ScoredCandidate


class KYCEntityMatcher:
    """Hybrid retrieval -> (learned rerank) -> structured verification -> explainable score."""

    def __init__(
        self,
        customers: Sequence[CanonicalCustomer],
        embedder: Embedder,
        config: MatcherConfig | None = None,
        reranker: LearnedReranker | None = None,
    ) -> None:
        self.config = config or MatcherConfig()
        self.customers = list(customers)
        self.signals = SignalStore(self.customers, embedder)
        self.scorer = CandidateScorer(self.customers, self.config, reranker)
        self.system = "full" if reranker is not None else "hybrid_structured"

    def match(self, document: CanonicalKYCDocument, top_n: int = 5) -> list[KYCMatchResult]:
        identity = QueryIdentity.from_document(document)
        (sig,) = self.signals.compute([identity.name.canonical])
        scored = self.scorer.score(self.system, identity, sig)[:top_n]
        return [self._to_result(c, identity) for c in scored]

    def _confidence(self, score: float) -> Confidence:
        cfg = self.config
        if score >= cfg.match_threshold + cfg.high_confidence_margin:
            return "high"
        return "medium" if score >= cfg.match_threshold else "low"

    def _to_result(self, c: ScoredCandidate, identity: QueryIdentity) -> KYCMatchResult:
        customer = self.customers[c.customer_idx]
        reasons: list[str] = []
        contradictions: list[str] = []
        qn = identity.name
        if qn.canonical == customer.name.canonical:
            reasons.append("name_exact")
        elif any(qn.canonical == a.canonical for a in customer.alternate_names):
            reasons.append("name_matches_alternate_name")
        elif c.fuzzy >= 0.85:
            reasons.append(f"name_close_spelling (similarity {c.fuzzy:.2f})")
        elif c.name_score < 0.5:
            contradictions.append(f"name_low_similarity ({c.name_score:.2f})")
        if qn.reordered:
            reasons.append("query_name_was_reordered_last_first")
        if qn.has_initials:
            reasons.append("query_name_contains_initials")
        if c.structured is not None:
            reasons += c.structured.reasons
            contradictions += c.structured.contradictions
        return KYCMatchResult(
            candidate_id=customer.customer.customer_id,
            lexical_score=c.bm25,
            semantic_score=c.dense,
            reranker_score=c.reranker,
            structured_match_score=c.structured.score if c.structured else None,
            final_score=c.final_score,
            is_match=c.final_score >= self.config.match_threshold,
            confidence=self._confidence(c.final_score),
            match_reasons=reasons,
            contradictory_evidence=contradictions,
            signals={
                "bm25": c.bm25,
                "bm25_char": c.bm25_char,
                "fuzzy": c.fuzzy,
                "dense": c.dense,
                "hybrid": c.hybrid,
            },
        )
