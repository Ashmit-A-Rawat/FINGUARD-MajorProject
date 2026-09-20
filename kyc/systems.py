"""The compared KYC systems, all built from the same signals so ablations are like-for-like."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from data_pipeline.consolidation.models import CanonicalCustomer
from kyc.lexical.exact import ExactIndex
from kyc.models import MatcherConfig, QueryIdentity
from kyc.reranking.features import build_feature_row
from kyc.reranking.reranker import LearnedReranker
from kyc.scoring.structured import StructuredVerification, verify_structured
from kyc.signals import QuerySignals

SYSTEMS = (
    "exact",
    "fuzzy",
    "bm25",
    "bm25_char",
    "dense",
    "hybrid",
    "hybrid_reranker",
    "hybrid_structured",
    "full",
)
_FIRST_STAGE = {
    "fuzzy": "fuzzy",
    "bm25": "bm25",
    "bm25_char": "bm25_char",
    "dense": "dense",
    "hybrid": "hybrid",
    "hybrid_reranker": "hybrid",
    "hybrid_structured": "hybrid",
    "full": "hybrid",
}


@dataclass
class ScoredCandidate:
    customer_idx: int
    final_score: float
    name_score: float  # the name-side score that entered final_score
    bm25: float
    bm25_char: float
    dense: float
    fuzzy: float
    hybrid: float
    reranker: float | None
    structured: StructuredVerification | None


class CandidateScorer:
    def __init__(
        self,
        customers: Sequence[CanonicalCustomer],
        config: MatcherConfig,
        reranker: LearnedReranker | None = None,
    ) -> None:
        self.customers = list(customers)
        self.config = config
        self.reranker = reranker
        self._exact = ExactIndex(self.customers)

    def hybrid(self, sig: QuerySignals) -> np.ndarray:
        a = self.config.hybrid_alpha
        result: np.ndarray = a * sig.bm25 + (1 - a) * sig.dense
        return result

    def _top_k(self, scores: np.ndarray) -> np.ndarray:
        k = min(self.config.top_k, len(scores))
        part = np.argpartition(-scores, k - 1)[:k]
        return part[np.argsort(-scores[part], kind="stable")]

    def feature_matrix(
        self, identity: QueryIdentity, sig: QuerySignals, idx: np.ndarray, hybrid: np.ndarray
    ) -> np.ndarray:
        rows = [
            build_feature_row(
                identity.name,
                self.customers[i],
                (sig.bm25[i], sig.bm25_char[i], sig.dense[i], sig.fuzzy[i], hybrid[i]),
                rank,
            )
            for rank, i in enumerate(idx, start=1)
        ]
        return np.vstack(rows)

    def training_candidates(
        self, identity: QueryIdentity, sig: QuerySignals
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-K hybrid candidates and their feature matrix (for fitting the reranker)."""
        hybrid = self.hybrid(sig)
        idx = self._top_k(hybrid)
        return idx, self.feature_matrix(identity, sig, idx, hybrid)

    def score(
        self, system: str, identity: QueryIdentity, sig: QuerySignals
    ) -> list[ScoredCandidate]:
        if system not in SYSTEMS:
            raise ValueError(f"unknown system {system!r}")
        hybrid = self.hybrid(sig)
        if system == "exact":
            idx = np.array(
                self._exact.lookup(identity.name.canonical, identity.date_of_birth), dtype=np.int64
            )
        else:
            first_stage = {
                "fuzzy": sig.fuzzy,
                "bm25": sig.bm25,
                "bm25_char": sig.bm25_char,
                "dense": sig.dense,
                "hybrid": hybrid,
            }[_FIRST_STAGE[system]]
            idx = self._top_k(first_stage)

        rerank: np.ndarray | None = None
        if system in ("hybrid_reranker", "full"):
            if self.reranker is None:
                raise RuntimeError(f"system {system!r} needs a fitted reranker")
            rerank = self.reranker.predict(self.feature_matrix(identity, sig, idx, hybrid))

        cfg = self.config
        weight_total = (cfg.weight_name + cfg.weight_structured) or 1.0
        out: list[ScoredCandidate] = []
        for pos, i in enumerate(idx):
            i = int(i)
            structured = (
                verify_structured(identity, self.customers[i], cfg)
                if system in ("hybrid_structured", "full")
                else None
            )
            reranker_score = float(rerank[pos]) if rerank is not None else None
            if system == "exact":
                name_score = final = 1.0
            elif reranker_score is not None:
                name_score = reranker_score
            else:
                name_score = float(first_stage[i])
            if structured is not None:
                final = (
                    cfg.weight_name * name_score + cfg.weight_structured * structured.score
                ) / weight_total
            elif system != "exact":
                final = name_score
            out.append(
                ScoredCandidate(
                    i,
                    final,
                    name_score,
                    float(sig.bm25[i]),
                    float(sig.bm25_char[i]),
                    float(sig.dense[i]),
                    float(sig.fuzzy[i]),
                    float(hybrid[i]),
                    reranker_score,
                    structured,
                )
            )
        out.sort(key=lambda c: -c.final_score)
        return out
