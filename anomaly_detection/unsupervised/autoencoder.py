"""Autoencoder anomaly detector: trained WITHOUT labels; score = reconstruction error."""

import copy

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from anomaly_detection.base import AnomalyModel, feature_matrix
from anomaly_detection.dataset import FeatureTable
from anomaly_detection.neural.common import DEVICE, fit_scaler, scale, set_seed
from anomaly_detection.temporal.splits import TemporalSplit


class AutoencoderModel(AnomalyModel):
    name, family, uses_labels = "autoencoder", "unsupervised", False

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
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, d),
        ).to(DEVICE)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        rng = np.random.default_rng(self._seed)
        best_state, best, stale = copy.deepcopy(net.state_dict()), float("inf"), 0
        for _ in range(self._max_epochs):
            net.train()
            order = rng.permutation(len(xs))
            for start in range(0, len(order), 512):
                batch = xs[order[start : start + 512]]
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(net(batch), batch)
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            net.eval()
            with torch.no_grad():  # unlabeled validation reconstruction loss: no labels used
                val_loss = float(nn.functional.mse_loss(net(xv), xv))
            self.history.append(val_loss)
            if val_loss < best - 1e-6:
                best, best_state, stale = val_loss, copy.deepcopy(net.state_dict()), 0
            else:
                stale += 1
                if stale >= 5:
                    break
        net.load_state_dict(best_state)
        net.eval()

    def score(self, table: FeatureTable, idx: np.ndarray) -> np.ndarray:
        assert self._net is not None and self._scaler is not None
        x = torch.tensor(scale(self._scaler, feature_matrix(table, idx, "engineered")))
        with torch.no_grad():
            errors: np.ndarray = ((self._net(x) - x) ** 2).mean(dim=1).numpy()
        return errors
