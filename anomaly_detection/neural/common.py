"""Shared PyTorch helpers (CPU, seeded)."""

import copy
from collections.abc import Callable

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch import nn

DEVICE = torch.device("cpu")  # small models; CPU keeps runs reproducible
# One thread: reproducible training, and consistent with the process-wide OpenMP limit
# (see backend/app/core/runtime.py for the clash this avoids).
torch.set_num_threads(1)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def fit_scaler(matrix: np.ndarray) -> StandardScaler:
    return StandardScaler().fit(matrix)


def scale(scaler: StandardScaler, matrix: np.ndarray) -> np.ndarray:
    scaled: np.ndarray = np.clip(scaler.transform(matrix), -10, 10).astype(np.float32)
    return scaled


def train_supervised(
    model: nn.Module,
    forward: Callable[[np.ndarray], torch.Tensor],
    train_idx: np.ndarray,
    y_train: np.ndarray,
    predict_val: Callable[[], np.ndarray],
    y_val: np.ndarray,
    *,
    max_epochs: int,
    patience: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> list[float]:
    """Train with class-balanced BCE; early-stop on validation PR-AUC; restore the best epoch.

    ``forward(batch_positions)`` returns logits for those rows. Returns the val PR-AUC history.
    ``y_train`` is aligned with ``train_idx`` (row i of y_train belongs to train_idx[i]).
    """
    rng = np.random.default_rng(seed)
    pos_weight = torch.tensor(float((y_train == 0).sum() / max((y_train == 1).sum(), 1)))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    y_all = torch.tensor(y_train, dtype=torch.float32)
    best_state, best_ap, stale, history = copy.deepcopy(model.state_dict()), -1.0, 0, []
    for _ in range(max_epochs):
        model.train()
        order = rng.permutation(len(train_idx))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(forward(train_idx[batch]), y_all[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            ap = float(average_precision_score(y_val, predict_val()))
        history.append(ap)
        if ap > best_ap + 1e-5:
            best_ap, best_state, stale = ap, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return history
