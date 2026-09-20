"""Learned reranker: gradient-boosted trees over name-pair features, trained on our benchmark."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class LearnedReranker:
    """Outputs P(same entity | name features). Fit on the train split only."""

    def __init__(self, seed: int = 0) -> None:
        self._model = HistGradientBoostingClassifier(
            max_depth=3, max_iter=150, learning_rate=0.1, random_state=seed
        )
        self._fitted = False

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LearnedReranker":
        if len(set(labels.tolist())) < 2:
            raise ValueError("reranker training needs both positive and negative examples")
        self._model.fit(features, labels)
        self._fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("reranker used before fit()")
        proba: np.ndarray = self._model.predict_proba(features)[:, 1]
        return proba
