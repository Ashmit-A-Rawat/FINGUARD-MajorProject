"""Anomaly scoring service for the Auditor agent.

Trains an XGBoost detector ONCE on the earliest 60% of the store's history (labels come from
analyst-confirmed history; here, the synthetic generator's labels) and chooses the alert threshold
on the next 20%. A transaction inside the training period gets ``in_training_period=True`` because
its score is then optimistic; evaluations must use cases from the held-out period.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from agents.contracts import AnomalyFinding
from anomaly_detection.features.builder import FeatureFrame, build_features
from anomaly_detection.temporal.splits import temporal_split
from data_pipeline.consolidation.store import ConsolidatedStore
from evaluation.metrics.anomaly import f1_optimal_threshold

TOP_DRIVERS = 3


class AnomalyService:
    def __init__(self, store: ConsolidatedStore, labels: pd.DataFrame, seed: int = 0) -> None:
        """``labels``: columns transaction_id, is_anomaly, episode_id (training targets only)."""
        transactions = pd.DataFrame(
            [t.model_dump() for txs in store.transactions_by_customer.values() for t in txs]
        )
        transactions = transactions.sort_values(["timestamp", "transaction_id"], kind="stable")
        self._tx = transactions.reset_index(drop=True)
        customers = pd.DataFrame([c.customer.model_dump() for c in store.customers.values()])
        self._features: FeatureFrame = build_features(self._tx, customers)
        info = labels.set_index("transaction_id").loc[self._tx["transaction_id"]]
        y = info["is_anomaly"].astype(str).eq("True").to_numpy().astype(int)
        timestamps = self._tx["timestamp"].to_numpy().astype("datetime64[ns]")
        split = temporal_split(timestamps, info["episode_id"].to_numpy(dtype=str))
        matrix = self._features.frame.to_numpy(np.float32)
        self._columns = list(self._features.frame.columns)
        self._model = xgb.XGBClassifier(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=float((y[split.train] == 0).sum() / max(y[split.train].sum(), 1)),
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=30,
            random_state=seed,
            n_jobs=1,
        )
        self._model.fit(
            matrix[split.train],
            y[split.train],
            eval_set=[(matrix[split.val], y[split.val])],
            verbose=False,
        )
        self.threshold = f1_optimal_threshold(
            y[split.val], self._model.predict_proba(matrix[split.val])[:, 1]
        )
        self.trained_until = pd.Timestamp(split.boundaries[0])
        self._row = {tid: i for i, tid in enumerate(self._tx["transaction_id"])}
        self._matrix = matrix

    def score(self, transaction_id: str) -> AnomalyFinding:
        i = self._row[transaction_id]
        row = self._matrix[i : i + 1]
        probability = float(self._model.predict_proba(row)[0, 1])
        contributions = self._model.get_booster().predict(xgb.DMatrix(row), pred_contribs=True)[0][
            :-1
        ]
        order = np.argsort(-np.abs(contributions))[:TOP_DRIVERS]
        return AnomalyFinding(
            transaction_id=transaction_id,
            probability=probability,
            flagged=probability >= self.threshold,
            threshold=float(self.threshold),
            top_drivers=[(self._columns[j], round(float(contributions[j]), 3)) for j in order],
            in_training_period=bool(self._tx["timestamp"].iloc[i] < self.trained_until),
            model="xgboost",
        )
