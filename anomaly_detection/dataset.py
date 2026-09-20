"""Load a validated dataset into a feature table for anomaly detection."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from anomaly_detection.features.builder import FeatureFrame, build_features
from data_pipeline.pipeline import run_pipeline

logger = logging.getLogger(__name__)


@dataclass
class FeatureTable:
    features: FeatureFrame
    labels: np.ndarray  # 1 = injected anomaly (evaluation/training target only)
    anomaly_type: np.ndarray
    episode_id: np.ndarray
    timestamps: np.ndarray
    customer_codes: np.ndarray
    transaction_ids: np.ndarray

    @property
    def n(self) -> int:
        return len(self.labels)


def load_feature_table(data_dir: Path, seq_len: int = 20) -> FeatureTable:
    """Ingest through the Phase 3 pipeline (validated, quarantined) then build features."""
    result = run_pipeline(data_dir)
    if result.quarantine:
        raise RuntimeError(f"{len(result.quarantine)} rows were quarantined; refusing to train")
    transactions = pd.DataFrame(
        [t.model_dump() for txs in result.store.transactions_by_customer.values() for t in txs]
    )
    transactions = transactions.sort_values(["timestamp", "transaction_id"], kind="stable")
    transactions = transactions.reset_index(drop=True)
    customers = pd.DataFrame([c.customer.model_dump() for c in result.store.customers.values()])
    labels = pd.read_csv(data_dir / "transaction_labels.csv", keep_default_na=False)
    labels = labels.set_index("transaction_id").loc[transactions["transaction_id"]]
    features = build_features(transactions, customers, seq_len)
    logger.info("features: %d rows x %d columns", len(transactions), features.frame.shape[1])
    return FeatureTable(
        features=features,
        labels=labels["is_anomaly"].astype(str).eq("True").to_numpy().astype(np.int8),
        anomaly_type=labels["anomaly_type"].to_numpy(dtype=str),
        episode_id=labels["episode_id"].to_numpy(dtype=str),
        timestamps=transactions["timestamp"].to_numpy().astype("datetime64[ns]"),
        customer_codes=features.customer_codes,
        transaction_ids=transactions["transaction_id"].to_numpy(dtype=str),
    )
