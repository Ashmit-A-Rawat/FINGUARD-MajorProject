"""Normal customer behaviour: per-customer profiles and baseline transaction generation.

Rows are first produced in a compact "raw" form (integer customer index, counterparty number)
so the anomaly injector can edit/insert rows in the same representation; ``materialize``
converts to the public transaction columns afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_pipeline.synthetic.config import GeneratorConfig
from data_pipeline.synthetic.reference_data import (
    ACCOUNT_TYPE_MULTIPLIER,
    CURRENCY_BY_COUNTRY,
    OCCUPATIONS,
    UNITS_PER_USD,
)

TX_TYPES = np.array(["transfer_out", "payment", "withdrawal", "deposit", "transfer_in"])
TX_TYPE_P = np.array([0.35, 0.30, 0.10, 0.10, 0.15])
OUTGOING = ("transfer_out", "payment", "withdrawal")
CHANNELS = np.array(["mobile", "web", "branch", "atm", "pos"])
CHANNEL_P = np.array([0.40, 0.25, 0.10, 0.10, 0.15])
FOREIGN_SHARE = 0.10  # share of baseline counterparties located abroad
NEW_COUNTERPARTY_SHARE = 0.15  # baseline chance a counterparty is outside the regular set

RAW_COLUMNS = ["cust_idx", "ts", "amount", "ttype", "cp_num", "cp_country", "channel"]


@dataclass
class Profiles:
    customer_id: np.ndarray
    account: np.ndarray
    home_country: np.ndarray
    currency: np.ndarray
    median_amount: np.ndarray  # local currency
    sigma: np.ndarray
    pref_hour: np.ndarray
    start_offset_days: np.ndarray
    regular_counterparties: list[np.ndarray]
    counterparty_pool: int

    @property
    def n(self) -> int:
        return len(self.customer_id)


def build_profiles(
    customers: pd.DataFrame, cfg: GeneratorConfig, rng: np.random.Generator
) -> Profiles:
    n = len(customers)
    home = customers["country"].to_numpy(dtype=str)
    currency = np.array([CURRENCY_BY_COUNTRY[c] for c in home])
    base_usd = np.array([OCCUPATIONS[o] for o in customers["occupation"]])
    mult = np.array([ACCOUNT_TYPE_MULTIPLIER[a] for a in customers["account_type"]])
    fx = np.array([UNITS_PER_USD[c] for c in currency])
    open_dates = pd.to_datetime(customers["account_open_date"])
    offset = (open_dates - pd.Timestamp(cfg.window_start)).dt.days.to_numpy()
    pool = max(1000, 5 * n)
    return Profiles(
        customer_id=customers["customer_id"].to_numpy(dtype=str),
        account=np.array(["ACC-" + c.split("-")[1] for c in customers["customer_id"]]),
        home_country=home,
        currency=currency,
        median_amount=base_usd * mult * fx * rng.lognormal(0.0, 0.3, n),
        sigma=rng.uniform(0.3, 0.7, n),
        pref_hour=np.clip(rng.normal(13.0, 3.0, n), 8.0, 19.0),
        start_offset_days=np.clip(offset, 0, cfg.window_days - 30),
        regular_counterparties=[rng.integers(0, pool, int(rng.integers(3, 13))) for _ in range(n)],
        counterparty_pool=pool,
    )


def to_timestamps(cfg: GeneratorConfig, seconds: np.ndarray) -> np.ndarray:
    base = np.datetime64(cfg.window_start.isoformat(), "s")
    return (base + seconds.astype("int64").astype("timedelta64[s]")).astype("datetime64[ns]")


def generate_normal(
    profiles: Profiles, n_tx: int, cfg: GeneratorConfig, rng: np.random.Generator
) -> pd.DataFrame:
    weights = rng.lognormal(0.0, 0.7, profiles.n)
    counts = rng.multinomial(n_tx, weights / weights.sum())
    foreign = np.array(sorted({c for c in profiles.home_country}))
    parts: list[pd.DataFrame] = []
    for i, k in enumerate(counts):
        if k == 0:
            continue
        start = int(profiles.start_offset_days[i])
        day = rng.integers(start, cfg.window_days, k)
        hour = rng.normal(profiles.pref_hour[i], 2.5, k) % 24
        seconds = day * 86400 + (hour * 3600).astype("int64") + rng.integers(0, 60, k)
        ttype = rng.choice(TX_TYPES, k, p=TX_TYPE_P)
        amount = rng.lognormal(np.log(profiles.median_amount[i]), profiles.sigma[i], k)
        regular = profiles.regular_counterparties[i]
        cp = np.where(
            rng.random(k) < NEW_COUNTERPARTY_SHARE,
            rng.integers(0, profiles.counterparty_pool, k),
            regular[rng.integers(0, len(regular), k)],
        )
        cp_country = np.where(
            rng.random(k) < FOREIGN_SHARE, rng.choice(foreign, k), profiles.home_country[i]
        )
        channel = rng.choice(CHANNELS, k, p=CHANNEL_P)
        channel = np.where(ttype == "withdrawal", rng.choice(["atm", "branch"], k), channel)
        channel = np.where(ttype == "payment", rng.choice(["pos", "web", "mobile"], k), channel)
        parts.append(
            pd.DataFrame(
                {
                    "cust_idx": np.full(k, i, dtype="int64"),
                    "ts": to_timestamps(cfg, seconds),
                    "amount": amount,
                    "ttype": ttype,
                    "cp_num": cp.astype("int64"),
                    "cp_country": cp_country,
                    "channel": channel,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def materialize(raw: pd.DataFrame, profiles: Profiles) -> pd.DataFrame:
    """Convert raw rows into public transaction columns (labels carried through untouched)."""
    idx = raw["cust_idx"].to_numpy()
    ttype = raw["ttype"].to_numpy(dtype=str)
    outgoing = np.isin(ttype, OUTGOING)
    prefix = np.where(ttype == "payment", "MRC-", "EXT-")
    counterparty = np.char.add(prefix, np.char.zfill(raw["cp_num"].to_numpy().astype(str), 7))
    counterparty = np.where(ttype == "withdrawal", "CASH", counterparty)
    own = profiles.account[idx]
    home = profiles.home_country[idx]
    cp_country = np.where(ttype == "withdrawal", home, raw["cp_country"].to_numpy(dtype=str))
    out = pd.DataFrame(
        {
            "customer_id": profiles.customer_id[idx],
            "timestamp": raw["ts"].to_numpy(),
            "amount": np.round(np.maximum(raw["amount"].to_numpy(), 1.0), 2),
            "currency": profiles.currency[idx],
            "transaction_type": ttype,
            "sender": np.where(outgoing, own, counterparty),
            "receiver": np.where(outgoing, counterparty, own),
            "sender_country": np.where(outgoing, home, cp_country),
            "receiver_country": np.where(outgoing, cp_country, home),
            "channel": raw["channel"].to_numpy(dtype=str),
        }
    )
    for label_col in ("anomaly_type", "episode_id"):
        if label_col in raw.columns:
            out[label_col] = raw[label_col].to_numpy(dtype=str)
    return out
