from anomaly_detection.base import AnomalyModel
from anomaly_detection.neural.mlp import MLPModel
from anomaly_detection.neural.temporal import TemporalModel
from anomaly_detection.supervised.models import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
)
from anomaly_detection.unsupervised.autoencoder import AutoencoderModel
from anomaly_detection.unsupervised.isolation_forest import IsolationForestModel


def create_models(seed: int) -> dict[str, AnomalyModel]:
    models: list[AnomalyModel] = [
        LogisticRegressionModel(seed),
        RandomForestModel(seed),
        XGBoostModel(seed),
        XGBoostModel(seed, feature_set="basic"),
        IsolationForestModel(seed),
        AutoencoderModel(seed),
        MLPModel(seed),
        TemporalModel(seed, use_engineered_head=False),
        TemporalModel(seed, use_engineered_head=True),
    ]
    return {m.name: m for m in models}
