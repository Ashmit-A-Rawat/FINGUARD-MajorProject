"""Shared, read-only services for the agents, built once."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from agents.audit import Clock, utc_now
from agents.auditor.anomaly_service import AnomalyService
from data_pipeline.consolidation.store import ConsolidatedStore
from data_pipeline.pipeline import run_pipeline
from knowledge_base.embeddings.embedder import Embedder
from knowledge_base.service import KnowledgeBase, SourceSpec
from kyc.entity_matcher import KYCEntityMatcher
from llm.inference.base import LLMProvider
from reconciliation.service import ReconciliationService


@dataclass
class AgentContext:
    store: ConsolidatedStore
    kyc_matcher: KYCEntityMatcher
    anomaly: AnomalyService
    reconciliation: ReconciliationService
    knowledge_base: KnowledgeBase
    llm: LLMProvider
    as_of: pd.Timestamp  # end of the loaded data; used for staleness rules
    clock: Clock = utc_now


def build_context(
    data_dir: Path,
    embedder: Embedder,
    llm: LLMProvider,
    knowledge_dir: Path = Path("knowledge_base/documents"),
    clock: Clock = utc_now,
) -> AgentContext:
    """Ingest through the validated pipeline and construct every engine once."""
    result = run_pipeline(data_dir)
    if result.quarantine:
        raise RuntimeError(f"{len(result.quarantine)} rows were quarantined; refusing to start")
    store = result.store
    labels = pd.read_csv(data_dir / "transaction_labels.csv", keep_default_na=False)
    txs = [t for v in store.transactions_by_customer.values() for t in v]
    return AgentContext(
        store=store,
        kyc_matcher=KYCEntityMatcher(list(store.customers.values()), embedder),
        anomaly=AnomalyService(store, labels),
        reconciliation=ReconciliationService(),
        knowledge_base=KnowledgeBase.build([SourceSpec(knowledge_dir)], embedder),
        llm=llm,
        as_of=pd.Timestamp(max(t.timestamp for t in txs)),
        clock=clock,
    )
