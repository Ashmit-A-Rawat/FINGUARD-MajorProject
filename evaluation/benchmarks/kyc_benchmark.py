"""KYC entity-resolution benchmark, derived from the synthetic generator's ground truth.

Task: given a KYC document (query), find the customer-master records of the SAME entity.
Gold = every customer sharing the query customer's entity_id (so duplicate customers are
gold; namesakes with the same name but a different person are NOT gold: they are the
hard negatives). Train/dev/test are split by ENTITY so no entity appears in two splits.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backend.app.schemas.domain import Customer, KYCRecord
from data_pipeline.consolidation.canonicalize import canonicalize_customer, canonicalize_kyc
from data_pipeline.consolidation.models import CanonicalCustomer, CanonicalKYCDocument
from data_pipeline.synthetic.config import GeneratorConfig
from data_pipeline.synthetic.customers import generate_customers

# Balanced so each hard category has enough queries (the production mix is mostly clean).
BENCHMARK_KYC_MIX = {
    "none": 0.20,
    "typo": 0.11,
    "abbreviation": 0.11,
    "transliteration": 0.11,
    "middle_name_diff": 0.11,
    "name_order_swap": 0.11,
    "address_mismatch": 0.13,
    "dob_conflict": 0.12,
}
SPLIT_FRACTIONS = (("train", 0.5), ("dev", 0.2), ("test", 0.3))


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str  # document_id
    customer_id: str
    document: CanonicalKYCDocument
    variation_type: str
    gold_customer_ids: frozenset[str]
    namesake_ids: frozenset[str]  # customers with the same name but a different entity
    split: str

    @property
    def has_duplicate_entity(self) -> bool:
        return len(self.gold_customer_ids) > 1

    @property
    def has_namesake_negative(self) -> bool:
        return bool(self.namesake_ids)


@dataclass
class KYCBenchmark:
    customers: list[CanonicalCustomer]
    queries: list[BenchmarkQuery]

    def split(self, name: str) -> list[BenchmarkQuery]:
        return [q for q in self.queries if q.split == name]

    def composition(self) -> dict[str, Any]:
        by_variation: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for q in self.queries:
            by_variation[q.variation_type][q.split] += 1
        return {
            "n_customers": len(self.customers),
            "n_queries": len(self.queries),
            "queries_per_split": {s: len(self.split(s)) for s, _ in SPLIT_FRACTIONS},
            "queries_by_variation_and_split": {k: dict(v) for k, v in sorted(by_variation.items())},
            "queries_with_duplicate_entity": sum(q.has_duplicate_entity for q in self.queries),
            "queries_with_namesake_negative": sum(q.has_namesake_negative for q in self.queries),
        }


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.astype(object).where(df.notna(), None).to_dict("records")  # type: ignore[no-any-return]


def benchmark_from_tables(
    customers: pd.DataFrame,
    kyc_records: pd.DataFrame,
    kyc_labels: pd.DataFrame,
    entity_labels: pd.DataFrame,
    seed: int,
) -> KYCBenchmark:
    canonical_customers = [
        canonicalize_customer(Customer.model_validate(r)) for r in _records(customers)
    ]
    entity_of = dict(zip(entity_labels["customer_id"], entity_labels["entity_id"], strict=True))
    members: dict[str, set[str]] = defaultdict(set)
    for cid, eid in entity_of.items():
        members[eid].add(cid)
    namesakes: dict[str, set[str]] = defaultdict(set)
    for row in entity_labels[entity_labels["relation"] == "namesake_of"].itertuples():
        namesakes[row.customer_id].add(row.related_customer_id)
        namesakes[row.related_customer_id].add(row.customer_id)

    entity_ids = sorted(members)
    order = np.random.default_rng(seed).permutation(len(entity_ids))
    split_of: dict[str, str] = {}
    start = 0
    for name, fraction in SPLIT_FRACTIONS:
        stop = (
            len(entity_ids)
            if name == SPLIT_FRACTIONS[-1][0]
            else start + round(fraction * len(entity_ids))
        )
        for i in order[start:stop]:
            split_of[entity_ids[int(i)]] = name
        start = stop

    variation_of = dict(zip(kyc_labels["document_id"], kyc_labels["variation_type"], strict=True))
    queries: list[BenchmarkQuery] = []
    for record in _records(kyc_records):
        document = canonicalize_kyc(KYCRecord.model_validate(record))
        cid = document.record.customer_id
        gold = frozenset(members[entity_of[cid]])
        near = frozenset(set().union(*(namesakes[g] for g in gold)) - gold)
        queries.append(
            BenchmarkQuery(
                query_id=document.record.document_id,
                customer_id=cid,
                document=document,
                variation_type=variation_of[document.record.document_id],
                gold_customer_ids=gold,
                namesake_ids=near,
                split=split_of[entity_of[cid]],
            )
        )
    return KYCBenchmark(canonical_customers, queries)


def build_kyc_benchmark(n_customers: int = 4000, seed: int = 42) -> KYCBenchmark:
    cfg = GeneratorConfig(
        seed=seed,
        n_customers=n_customers,
        kyc_variation_mix=BENCHMARK_KYC_MIX,
        duplicate_rate=0.06,
        namesake_rate=0.06,
    )
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    customers, kyc, kyc_labels, entity_labels = generate_customers(cfg, rng)
    return benchmark_from_tables(customers, kyc, kyc_labels, entity_labels, seed)
