import numpy as np
from sklearn.ensemble import IsolationForest

from anomaly_detection.base import AnomalyModel, feature_matrix
from anomaly_detection.dataset import FeatureTable
from anomaly_detection.temporal.splits import TemporalSplit


class IsolationForestModel(AnomalyModel):
    """Fit WITHOUT labels on the training period."""

    name, family, uses_labels = "isolation_forest", "unsupervised", False

    def __init__(self, seed: int) -> None:
        self._model = IsolationForest(
            n_estimators=300, max_samples=512, random_state=seed, n_jobs=-1
        )

    def fit(self, table: FeatureTable, split: TemporalSplit) -> None:
        self._model.fit(feature_matrix(table, split.train, "engineered"))

    def score(self, table: FeatureTable, idx: np.ndarray) -> np.ndarray:
        scores: np.ndarray = -self._model.score_samples(feature_matrix(table, idx, "engineered"))
        return scores
