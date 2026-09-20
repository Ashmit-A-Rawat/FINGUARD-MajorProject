"""Supervised MLP on engineered features."""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from anomaly_detection.base import AnomalyModel, feature_matrix
from anomaly_detection.dataset import FeatureTable
from anomaly_detection.neural.common import DEVICE, fit_scaler, scale, set_seed, train_supervised
from anomaly_detection.temporal.splits import TemporalSplit


class MLPModel(AnomalyModel):
    name, family = "mlp", "neural"

    def __init__(self, seed: int, max_epochs: int = 40) -> None:
        self._seed, self._max_epochs = seed, max_epochs
        self._net: nn.Module | None = None
        self._scaler: StandardScaler | None = None
        self.history: list[float] = []

    def fit(self, table: FeatureTable, split: TemporalSplit) -> None:
        set_seed(self._seed)
        x_train = feature_matrix(table, split.train, "engineered")
        self._scaler = fit_scaler(x_train)
        xs = torch.tensor(scale(self._scaler, x_train))
        xv = torch.tensor(scale(self._scaler, feature_matrix(table, split.val, "engineered")))
        d = xs.shape[1]
        self._net = net = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        ).to(DEVICE)
        self.history = train_supervised(
            net,
            lambda b: net(xs[b]).squeeze(-1),
            np.arange(len(xs)),
            table.labels[split.train],
            lambda: net(xv).squeeze(-1).numpy(),
            table.labels[split.val],
            max_epochs=self._max_epochs,
            patience=5,
            lr=1e-3,
            batch_size=512,
            seed=self._seed,
        )

    def score(self, table: FeatureTable, idx: np.ndarray) -> np.ndarray:
        assert self._net is not None and self._scaler is not None
        x = torch.tensor(scale(self._scaler, feature_matrix(table, idx, "engineered")))
        with torch.no_grad():
            scores: np.ndarray = torch.sigmoid(self._net(x).squeeze(-1)).numpy()
        return scores
