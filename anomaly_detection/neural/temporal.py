"""Temporal anomaly model over each customer's recent transaction sequence.

    step inputs -> embeddings -> GRU over the last L transactions of the customer
    -> customer behavioural representation -> anomaly head

Why it exists: engineered history features (z-scores, window counts, "new counterparty") encode
behaviour by hand. This model gets only raw per-transaction attributes plus categorical embeddings
(counterparty hash bucket, country, type, channel) and inter-event times, and must learn
behaviour from the sequence. ``temporal_seq`` tests that alone; ``temporal_hybrid`` also feeds the
engineered features of the current transaction to the head. Sequences contain only rows at or
before the current transaction (see ``seq_index``), so there is no future leakage.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from anomaly_detection.base import AnomalyModel, FeatureSet, feature_matrix
from anomaly_detection.dataset import FeatureTable
from anomaly_detection.features.builder import CP_BUCKETS
from anomaly_detection.neural.common import DEVICE, fit_scaler, scale, set_seed, train_supervised
from anomaly_detection.temporal.splits import TemporalSplit

NUMERIC_STEP_COLUMNS = [
    "log_amount_usd",
    "hour_sin",
    "hour_cos",
    "is_night",
    "is_outgoing",
    "is_cross_border",
]


class _SequenceNet(nn.Module):
    def __init__(
        self, n_numeric: int, n_countries: int, n_engineered: int, hidden: int = 64
    ) -> None:
        super().__init__()
        self.emb_type = nn.Embedding(5, 4)
        self.emb_channel = nn.Embedding(5, 4)
        self.emb_country = nn.Embedding(n_countries, 8)
        self.emb_bucket = nn.Embedding(CP_BUCKETS, 16)
        step_dim = n_numeric + 3 + 4 + 4 + 8 + 16  # + dt_to_current, dt_prev, valid flag
        self.step_proj = nn.Sequential(nn.Linear(step_dim, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.engineered = (
            nn.Sequential(nn.Linear(n_engineered, 32), nn.ReLU()) if n_engineered else None
        )
        head_in = 2 * hidden + (32 if n_engineered else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def forward(
        self,
        numeric: torch.Tensor,
        cats: torch.Tensor,
        dts: torch.Tensor,
        engineered: torch.Tensor | None,
    ) -> torch.Tensor:
        steps = torch.cat(
            [
                numeric,
                dts,
                self.emb_type(cats[..., 0]),
                self.emb_channel(cats[..., 1]),
                self.emb_country(cats[..., 2]),
                self.emb_bucket(cats[..., 3]),
            ],
            dim=-1,
        )
        projected = self.step_proj(steps)
        output, _ = self.gru(projected)
        parts = [output[:, -1], projected[:, -1]]  # behaviour summary + current transaction
        if self.engineered is not None and engineered is not None:
            parts.append(self.engineered(engineered))
        logits: torch.Tensor = self.head(torch.cat(parts, dim=-1)).squeeze(-1)
        return logits


class TemporalModel(AnomalyModel):
    family = "neural"

    def __init__(self, seed: int, use_engineered_head: bool, max_epochs: int = 15) -> None:
        self.name = "temporal_hybrid" if use_engineered_head else "temporal_seq"
        self.feature_set: FeatureSet = "engineered" if use_engineered_head else "basic"
        self._seed, self._max_epochs, self._hybrid = seed, max_epochs, use_engineered_head
        self._net: _SequenceNet | None = None
        self._numeric_scaler: StandardScaler | None = None
        self._eng_scaler: StandardScaler | None = None
        self.history: list[float] = []

    def _tensors(self, table: FeatureTable) -> None:
        """Whole-table tensors; batches are gathered from these by position."""
        assert self._numeric_scaler is not None
        frame = table.features.frame
        self._numeric = torch.tensor(
            scale(self._numeric_scaler, frame[NUMERIC_STEP_COLUMNS].to_numpy(np.float32))
        )
        cats = table.features.categorical
        self._cats = torch.tensor(
            np.stack([cats["type"], cats["channel"], cats["cp_country"], cats["cp_bucket"]], axis=1)
        )
        self._seq = torch.tensor(table.features.seq_index)
        self._t = torch.tensor(table.timestamps.astype("int64") // 10**9, dtype=torch.float64)
        self._eng: torch.Tensor | None = None
        if self._hybrid:
            assert self._eng_scaler is not None
            self._eng = torch.tensor(scale(self._eng_scaler, frame.to_numpy(np.float32)))

    def _batch(
        self, idx: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        rows = torch.as_tensor(idx)
        seq = self._seq[rows]  # (B, L) global positions, -1 = padding
        valid = seq >= 0
        safe = seq.clamp(min=0)
        t_step = self._t[safe]
        dt_cur = torch.log1p((self._t[rows][:, None] - t_step).clamp(min=0)).float() * valid
        gaps = (t_step[:, 1:] - t_step[:, :-1]).clamp(min=0)
        both = valid[:, 1:] & valid[:, :-1]
        dt_prev = torch.cat(
            [torch.zeros_like(dt_cur[:, :1]), torch.log1p(gaps).float() * both], dim=1
        )
        dts = torch.stack([dt_cur, dt_prev, valid.float()], dim=-1)
        numeric = self._numeric[safe] * valid[..., None]
        engineered = self._eng[rows] if self._eng is not None else None
        return numeric, self._cats[safe], dts, engineered

    def _logits(self, idx: np.ndarray) -> torch.Tensor:
        assert self._net is not None
        logits: torch.Tensor = self._net(*self._batch(idx))
        return logits

    def fit(self, table: FeatureTable, split: TemporalSplit) -> None:
        set_seed(self._seed)
        frame = table.features.frame
        self._numeric_scaler = fit_scaler(
            frame[NUMERIC_STEP_COLUMNS].to_numpy(np.float32)[split.train]
        )
        if self._hybrid:
            self._eng_scaler = fit_scaler(feature_matrix(table, split.train, "engineered"))
        self._tensors(table)
        n_countries = int(table.features.categorical["cp_country"].max()) + 1
        self._net = net = _SequenceNet(
            len(NUMERIC_STEP_COLUMNS), n_countries, frame.shape[1] if self._hybrid else 0
        ).to(DEVICE)
        self.history = train_supervised(
            net,
            self._logits,
            split.train,
            table.labels[split.train],
            lambda: self._predict(split.val),
            table.labels[split.val],
            max_epochs=self._max_epochs,
            patience=4,
            lr=2e-3,
            batch_size=512,
            seed=self._seed,
        )

    def _predict(self, idx: np.ndarray) -> np.ndarray:
        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(idx), 4096):
                out.append(self._logits(idx[start : start + 4096]).numpy())
        return np.concatenate(out)

    def score(self, table: FeatureTable, idx: np.ndarray) -> np.ndarray:
        assert self._net is not None
        return np.asarray(1.0 / (1.0 + np.exp(-self._predict(idx))))
