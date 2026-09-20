"""Controlled behavioural-anomaly injection.

Two mechanisms:
* in-place edits of existing baseline rows (high_value, unusual_time, geo_change, new_counterparty)
* inserted "episodes" of new rows (burst, abnormal_velocity, repeated_transfers)

Every affected row gets ``anomaly_type`` (and ``episode_id`` for episodes). Labels are later split
into a separate file so they can never leak into model features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.synthetic.config import ANOMALY_TYPES, INSERTED_ANOMALY_TYPES, GeneratorConfig
from data_pipeline.synthetic.reference_data import UNUSUAL_COUNTRIES
from data_pipeline.synthetic.transactions import RAW_COLUMNS, Profiles, to_timestamps


def plan_anomaly_quotas(cfg: GeneratorConfig) -> dict[str, int]:
    """Rows to label per anomaly type; sums exactly to round(rate * n_transactions)."""
    total = round(cfg.anomaly_rate * cfg.n_transactions)
    quotas = {t: int(cfg.anomaly_mix[t] * total) for t in ANOMALY_TYPES}
    quotas["high_value"] += total - sum(quotas.values())  # rounding remainder
    return quotas


def count_inserted_rows(cfg: GeneratorConfig) -> int:
    quotas = plan_anomaly_quotas(cfg)
    return sum(quotas[t] for t in INSERTED_ANOMALY_TYPES)


def _episode(
    kind: str,
    cust: int,
    size: int,
    profiles: Profiles,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
    fresh_cp: int,
) -> pd.DataFrame:
    median = profiles.median_amount[cust]
    start = int(profiles.start_offset_days[cust])
    if kind == "repeated_transfers":
        day = rng.uniform(start + 1, cfg.window_days - 25)
        offsets = np.cumsum(rng.uniform(1, 3, size)) * 86400
        seconds = day * 86400 + offsets
        amounts = np.full(size, round(float(median * rng.uniform(2, 4)), -1) or 10.0)
        ttype = np.full(size, "transfer_out")
        cp = np.full(size, fresh_cp)
    else:
        day = rng.uniform(start + 1, cfg.window_days - 2)
        if kind == "burst":
            offsets = np.sort(rng.uniform(0, 600, size))
            amounts = median * rng.uniform(1.5, 4.0) * rng.uniform(0.9, 1.1, size)
            ttype = np.full(size, "transfer_out")
            cp = np.full(size, fresh_cp)
        else:  # abnormal_velocity
            offsets = np.sort(rng.uniform(0, 86400, size))
            amounts = median * rng.uniform(0.3, 1.5, size)
            ttype = rng.choice(["transfer_out", "payment"], size)
            cp = fresh_cp + np.arange(size)
        seconds = day * 86400 + offsets
    return pd.DataFrame(
        {
            "cust_idx": np.full(size, cust, dtype="int64"),
            "ts": to_timestamps(cfg, seconds),
            "amount": amounts,
            "ttype": ttype,
            "cp_num": cp.astype("int64"),
            "cp_country": profiles.home_country[cust],
            "channel": rng.choice(["web", "mobile"], size),
        }
    )[RAW_COLUMNS]


def inject_anomalies(
    raw: pd.DataFrame, profiles: Profiles, cfg: GeneratorConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return (raw rows incl. anomalies, realised row count per anomaly type)."""
    quotas = plan_anomaly_quotas(cfg)
    raw = raw.copy()
    raw["anomaly_type"] = ""
    raw["episode_id"] = ""
    realised = dict.fromkeys(ANOMALY_TYPES, 0)
    fresh = profiles.counterparty_pool  # ids >= pool were never used by baseline rows

    available = np.ones(len(raw), dtype=bool)
    is_transfer_out = (raw["ttype"] == "transfer_out").to_numpy()

    def pick(n: int, mask: np.ndarray) -> np.ndarray:
        candidates = np.flatnonzero(available & mask)
        chosen = rng.choice(candidates, min(n, len(candidates)), replace=False)
        available[chosen] = False
        return chosen

    everything = np.ones(len(raw), dtype=bool)
    for kind, mask in (
        ("geo_change", is_transfer_out),
        ("new_counterparty", is_transfer_out),
        ("high_value", is_transfer_out),
        ("unusual_time", everything),
    ):
        rows = pick(quotas[kind], mask)
        realised[kind] = len(rows)
        raw.loc[rows, "anomaly_type"] = kind
        if kind == "geo_change":
            raw.loc[rows, "cp_country"] = rng.choice(UNUSUAL_COUNTRIES, len(rows))
            raw.loc[rows, "channel"] = "web"
        elif kind == "new_counterparty":
            raw.loc[rows, "cp_num"] = fresh + np.arange(len(rows))
            raw.loc[rows, "amount"] = raw.loc[rows, "amount"] * rng.uniform(2, 4, len(rows))
            fresh += len(rows)
        elif kind == "high_value":
            raw.loc[rows, "amount"] = raw.loc[rows, "amount"] * rng.uniform(8, 30, len(rows))
        else:  # unusual_time: move to 01:00-04:59 on the same day
            secs = rng.integers(3600, 5 * 3600, len(rows))
            day_start = raw.loc[rows, "ts"].dt.normalize()
            raw.loc[rows, "ts"] = day_start + pd.to_timedelta(secs, unit="s")

    episodes: list[pd.DataFrame] = []
    size_range = {"burst": (5, 13), "abnormal_velocity": (15, 31), "repeated_transfers": (4, 9)}
    for kind in INSERTED_ANOMALY_TYPES:
        remaining, n_ep = quotas[kind], 0
        while remaining > 0:
            lo, hi = size_range[kind]
            size = min(remaining, int(rng.integers(lo, hi)))
            cust = int(rng.integers(0, profiles.n))
            ep = _episode(kind, cust, size, profiles, cfg, rng, fresh)
            fresh += size
            ep["anomaly_type"] = kind
            ep["episode_id"] = f"EP-{kind[:4].upper()}-{n_ep:05d}"
            episodes.append(ep)
            remaining -= size
            realised[kind] += size
            n_ep += 1
    if episodes:
        raw = pd.concat([raw, *episodes], ignore_index=True)
    return raw, realised
