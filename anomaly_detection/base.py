"""Common interface so every model is trained, thresholded and timed identically."""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from anomaly_detection.dataset import FeatureTable
from anomaly_detection.temporal.splits import TemporalSplit

FeatureSet = Literal["engineered", "basic"]


def feature_matrix(table: FeatureTable, idx: np.ndarray, feature_set: FeatureSet) -> np.ndarray:
    frame = table.features.frame
    columns = table.features.basic_columns if feature_set == "basic" else list(frame.columns)
    matrix: np.ndarray = frame[columns].to_numpy(dtype=np.float32)[idx]
    return matrix


class AnomalyModel(ABC):
    name: str
    family: Literal["supervised", "unsupervised", "neural"]
    feature_set: FeatureSet = "engineered"
    uses_labels: bool = True

    @abstractmethod
    def fit(self, table: FeatureTable, split: TemporalSplit) -> None:
        """Train using ``split.train`` (and ``split.val`` for early stopping only)."""

    @abstractmethod
    def score(self, table: FeatureTable, idx: np.ndarray) -> np.ndarray:
        """Anomaly score per row of ``idx``; higher = more anomalous."""
