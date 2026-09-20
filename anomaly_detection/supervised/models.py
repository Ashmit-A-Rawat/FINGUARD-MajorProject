"""Supervised baselines: logistic regression, random forest, XGBoost."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from anomaly_detection.base import AnomalyModel, FeatureSet, feature_matrix
from anomaly_detection.dataset import FeatureTable
from anomaly_detection.temporal.splits import TemporalSplit


class LogisticRegressionModel(AnomalyModel):
    name, family = "logistic_regression", "supervised"

    def __init__(self, seed: int) -> None:
        self._model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed),
        )

    def fit(self, table: FeatureTable, split: TemporalSplit) -> None:
        self._model.fit(feature_matrix(table, split.train, "engineered"), table.labels[split.train])

    def score(self, table: FeatureTable, idx: np.ndarray) -> np.ndarray:
        proba: np.ndarray = self._model.predict_proba(feature_matrix(table, idx, "engineered"))[
            :, 1
        ]
        return proba


class RandomForestModel(AnomalyModel):
    name, family = "random_forest", "supervised"

    def __init__(self, seed: int) -> None:
        self._model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )

    def fit(self, table: FeatureTable, split: TemporalSplit) -> None:
        self._model.fit(feature_matrix(table, split.train, "engineered"), table.labels[split.train])

    def score(self, table: FeatureTable, idx: np.ndarray) -> np.ndarray:
        proba: np.ndarray = self._model.predict_proba(feature_matrix(table, idx, "engineered"))[
            :, 1
        ]
        return proba


class XGBoostModel(AnomalyModel):
    family = "supervised"

    def __init__(self, seed: int, feature_set: FeatureSet = "engineered") -> None:
        self.feature_set = feature_set
        self.name = "xgboost" if feature_set == "engineered" else "xgboost_basic_features"
        self._seed = seed
        self._model: XGBClassifier | None = None

    def fit(self, table: FeatureTable, split: TemporalSplit) -> None:
        y = table.labels[split.train]
        self._model = XGBClassifier(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)),
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=30,
            random_state=self._seed,
            n_jobs=-1,
        )
        self._model.fit(
            feature_matrix(table, split.train, self.feature_set),
            y,
            eval_set=[
                (feature_matrix(table, split.val, self.feature_set), table.labels[split.val])
            ],
            verbose=False,
        )

    def score(self, table: FeatureTable, idx: np.ndarray) -> np.ndarray:
        assert self._model is not None
        proba: np.ndarray = self._model.predict_proba(feature_matrix(table, idx, self.feature_set))[
            :, 1
        ]
        return proba
