"""Causal feature engineering for transaction anomaly detection.

LEAKAGE RULE: every feature of transaction t is computed from t itself and from the SAME
customer's transactions that come strictly before t in (timestamp, position) order. No feature
uses labels, and none looks at later rows. `tests/anomaly` verifies this by recomputing features
on a truncated dataset and requiring identical values for the surviving rows.

Amounts are converted to USD-equivalent with the illustrative constant FX table that ships with
the synthetic data (it is not market data).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_pipeline.synthetic.reference_data import UNITS_PER_USD

OUTGOING = ("transfer_out", "payment", "withdrawal")
TX_TYPES = ("transfer_out", "transfer_in", "payment", "withdrawal", "deposit")
CHANNELS = ("mobile", "web", "branch", "atm", "pos")
ACCOUNT_TYPES = ("personal", "business", "savings")
WINDOWS = {"10min": 600, "1h": 3600, "24h": 86400, "7d": 7 * 86400}
MIN_HISTORY = 5  # prior transactions needed before history statistics are trusted
CP_BUCKETS = 1024
_KEY_SHIFT = 2**32  # composite (customer, time) sort key


@dataclass
class FeatureFrame:
    frame: pd.DataFrame  # one row per transaction, same order as the input
    basic_columns: list[str]  # per-transaction attributes only (no history)
    categorical: dict[str, np.ndarray]  # integer codes for embeddings
    seq_index: np.ndarray  # (N, L) global row positions of the customer's last L rows; -1 = pad
    customer_codes: np.ndarray

    @property
    def engineered_columns(self) -> list[str]:
        return list(self.frame.columns)


def _group_start(codes_sorted: np.ndarray) -> np.ndarray:
    idx = np.arange(len(codes_sorted))
    new_group = np.r_[True, codes_sorted[1:] != codes_sorted[:-1]]
    start: np.ndarray = np.maximum.accumulate(np.where(new_group, idx, 0))
    return start


def _prior_sum(values: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Sum of ``values`` over earlier rows of the same group (exclusive)."""
    exclusive = np.cumsum(values) - values
    prior: np.ndarray = exclusive - exclusive[start]
    return prior


def _prior_count_by(keys: list[np.ndarray]) -> np.ndarray:
    """How many earlier rows (in current order) share the same key tuple."""
    frame = pd.DataFrame({f"k{i}": k for i, k in enumerate(keys)})
    counts: np.ndarray = frame.groupby(list(frame.columns), sort=False).cumcount().to_numpy()
    return counts


def build_features(tx: pd.DataFrame, customers: pd.DataFrame, seq_len: int = 20) -> FeatureFrame:
    """``tx`` must be sorted by timestamp. Returns features aligned with ``tx`` rows."""
    if not tx["timestamp"].is_monotonic_increasing:
        raise ValueError("transactions must be sorted by timestamp")
    n = len(tx)
    ts = tx["timestamp"].astype("datetime64[ns]")
    ts_sec = (ts.astype("int64") // 10**9).to_numpy()
    hour = (ts.dt.hour + ts.dt.minute / 60.0).to_numpy()

    cust = customers.set_index("customer_id")
    home = tx["customer_id"].map(cust["country"]).to_numpy(dtype=str)
    open_date = pd.to_datetime(tx["customer_id"].map(cust["account_open_date"]))
    account_type = tx["customer_id"].map(cust["account_type"]).to_numpy(dtype=str)
    ttype = tx["transaction_type"].to_numpy(dtype=str)
    outgoing = np.isin(ttype, OUTGOING)
    counterparty = np.where(outgoing, tx["receiver"], tx["sender"]).astype(str)
    cp_country = np.where(outgoing, tx["receiver_country"], tx["sender_country"]).astype(str)
    fx = tx["currency"].map(UNITS_PER_USD).to_numpy(dtype=float)
    amount_usd = tx["amount"].to_numpy(dtype=float) / fx
    log_amount = np.log1p(amount_usd)

    cust_codes, _ = pd.factorize(tx["customer_id"])
    order = np.lexsort((np.arange(n), ts_sec, cust_codes))  # by customer, then time
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n)

    cc, t_s = cust_codes[order], ts_sec[order]
    start = _group_start(cc)
    pos = np.arange(n)
    n_prior = pos - start
    keys = cc.astype(np.int64) * _KEY_SHIFT + t_s
    la, au, hr = log_amount[order], amount_usd[order], hour[order]

    enough = n_prior >= MIN_HISTORY
    safe_n = np.maximum(n_prior, 1)
    mean_la = _prior_sum(la, start) / safe_n
    var_la = np.maximum(_prior_sum(la**2, start) / safe_n - mean_la**2, 0.0)
    z_amount = np.where(enough, (la - mean_la) / np.maximum(np.sqrt(var_la), 0.25), 0.0)
    prior_max = pd.Series(la).groupby(cc).cummax().groupby(cc).shift(1).fillna(0.0).to_numpy()
    hour_dev = np.where(enough, np.abs(hr - _prior_sum(hr, start) / safe_n), 0.0)
    gap = np.where(n_prior > 0, t_s - np.r_[t_s[:1], t_s[:-1]], 30 * 86400)

    cols: dict[str, np.ndarray] = {}
    for name, seconds in WINDOWS.items():
        w_start = np.searchsorted(keys, keys - seconds, side="left")
        cols[f"log_count_{name}"] = np.log1p(pos - w_start)
    w24 = np.searchsorted(keys, keys - WINDOWS["24h"], side="left")
    exclusive_amount = np.cumsum(au) - au
    cols["log_sum_amount_24h"] = np.log1p(np.maximum(exclusive_amount - exclusive_amount[w24], 0.0))

    cp_codes = pd.factorize(counterparty[order])[0]
    country_codes = pd.factorize(cp_country[order])[0]
    cols["is_new_counterparty"] = (_prior_count_by([cc, cp_codes]) == 0).astype(float)
    cols["is_new_counterparty_country"] = (_prior_count_by([cc, country_codes]) == 0).astype(float)
    cols["log_prior_same_amount_to_counterparty"] = np.log1p(
        _prior_count_by([cc, cp_codes, np.round(au, 2)])
    )
    cols["log_prior_counterparty_count"] = np.log1p(_prior_count_by([cc, cp_codes]))
    cols["log_n_prior_tx"] = np.log1p(n_prior)
    cols["z_amount_vs_history"] = z_amount
    cols["amount_over_prior_max"] = np.where(enough, la - prior_max, 0.0)
    cols["hour_deviation"] = hour_dev
    cols["log_gap_since_prev_s"] = np.log1p(gap)

    frame_sorted = pd.DataFrame(cols)
    engineered = frame_sorted.iloc[inverse].reset_index(drop=True)

    basic: dict[str, np.ndarray] = {
        "log_amount_usd": log_amount,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "is_night": (hour < 6).astype(float),
        "is_outgoing": outgoing.astype(float),
        "is_cross_border": (cp_country != home).astype(float),
    }
    basic.update({f"type_{t}": (ttype == t).astype(float) for t in TX_TYPES})
    basic.update(
        {f"channel_{c}": (tx["channel"].to_numpy(dtype=str) == c).astype(float) for c in CHANNELS}
    )
    static: dict[str, np.ndarray] = {
        "log_account_age_days": np.log1p(
            np.maximum((ts - open_date).dt.days.to_numpy(dtype=float), 0)
        ),
    }
    static.update({f"account_{a}": (account_type == a).astype(float) for a in ACCOUNT_TYPES})

    frame = pd.concat([pd.DataFrame(basic), engineered, pd.DataFrame(static)], axis=1).astype(
        "float32"
    )

    seq_sorted = np.full((n, seq_len), -1, dtype=np.int64)
    for k in range(seq_len):
        back = seq_len - 1 - k
        source = pos - back
        valid = source >= start
        seq_sorted[:, k] = np.where(valid, order[np.maximum(source, 0)], -1)
    seq_index = seq_sorted[inverse]

    categorical = {
        "type": pd.Categorical(ttype, categories=TX_TYPES).codes.astype(np.int64),
        "channel": pd.Categorical(tx["channel"], categories=CHANNELS).codes.astype(np.int64),
        "cp_country": pd.factorize(cp_country, sort=True)[0].astype(np.int64),
        "cp_bucket": (pd.util.hash_array(counterparty) % CP_BUCKETS).astype(np.int64),
    }
    return FeatureFrame(frame, list(basic), categorical, seq_index, cust_codes)
