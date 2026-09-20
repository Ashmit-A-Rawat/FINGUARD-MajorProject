"""Ledger generation with controlled transaction<->ledger discrepancies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.synthetic.config import LEDGER_DISCREPANCIES, GeneratorConfig

CURRENCIES = ["USD", "GBP", "EUR", "INR", "AED", "SGD", "CAD", "AUD"]


def build_ledger(
    tx: pd.DataFrame, cfg: GeneratorConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ledger, ledger_labels). ``tx`` must already carry transaction_id/reference_id."""
    n = len(tx)
    delay = rng.exponential(1800, n) + np.where(
        rng.random(n) < 0.10, rng.uniform(6, 30, n) * 3600, 0
    )
    ledger = pd.DataFrame(
        {
            "transaction_id": tx["transaction_id"].to_numpy(),
            "posted_amount": tx["amount"].to_numpy().copy(),
            "currency": tx["currency"].to_numpy(dtype=str).copy(),
            "posting_timestamp": tx["timestamp"].to_numpy()
            + (delay * 1e9).astype("int64").astype("timedelta64[ns]"),
            "settlement_status": rng.choice(
                ["settled", "pending", "failed"], n, p=[0.95, 0.04, 0.01]
            ),
            "reference_id": tx["reference_id"].to_numpy(dtype=object).copy(),
            "_disc": "",
        }
    )

    total = round(cfg.ledger_discrepancy_rate * n)
    quotas = {d: int(cfg.ledger_discrepancy_mix[d] * total) for d in LEDGER_DISCREPANCIES}
    quotas["contradictory_amount"] += total - sum(quotas.values())
    chosen = rng.permutation(n)[:total]
    cursor = 0
    dropped: list[int] = []
    duplicates: list[pd.DataFrame] = []
    for kind in LEDGER_DISCREPANCIES:
        rows = chosen[cursor : cursor + quotas[kind]]
        cursor += quotas[kind]
        if len(rows) == 0:
            continue
        ledger.loc[rows, "_disc"] = kind
        if kind == "contradictory_amount":
            up = rng.random(len(rows)) < 0.5
            factor = np.where(
                up, rng.uniform(1.02, 1.5, len(rows)), rng.uniform(0.5, 0.98, len(rows))
            )
            ledger.loc[rows, "posted_amount"] = np.round(
                ledger.loc[rows, "posted_amount"] * factor, 2
            )
        elif kind == "missing_reference":
            ledger.loc[rows, "reference_id"] = None
        elif kind == "currency_mismatch":
            ledger.loc[rows, "currency"] = [
                str(rng.choice([c for c in CURRENCIES if c != cur]))
                for cur in ledger.loc[rows, "currency"]
            ]
        elif kind == "late_posting":
            late = (
                (rng.uniform(8, 20, len(rows)) * 86400e9).astype("int64").astype("timedelta64[ns]")
            )
            ledger.loc[rows, "posting_timestamp"] = ledger.loc[rows, "posting_timestamp"] + late
        elif kind == "duplicate_posting":
            dup = ledger.loc[rows].copy()
            dup["posting_timestamp"] = dup["posting_timestamp"] + pd.to_timedelta(
                rng.integers(1, 30, len(dup)), unit="s"
            )
            dup["_is_dup"] = True
            duplicates.append(dup)
        else:  # missing_ledger_entry
            dropped.extend(int(r) for r in rows)

    missing_tx = ledger.loc[dropped, "transaction_id"].tolist()
    ledger = ledger.drop(index=dropped)
    ledger["_is_dup"] = False
    ledger = pd.concat([ledger, *duplicates], ignore_index=True)
    ledger = ledger.sort_values("posting_timestamp", kind="stable").reset_index(drop=True)
    ledger.insert(0, "ledger_id", [f"LED-{i:08d}" for i in range(len(ledger))])
    ledger["is_synthetic"] = True

    flagged = ledger[(ledger["_disc"] != "") & (ledger["_disc"] != "duplicate_posting")]
    dup_rows = ledger[ledger["_is_dup"]]
    labels = pd.concat(
        [
            flagged[["transaction_id", "ledger_id", "_disc"]],
            dup_rows[["transaction_id", "ledger_id"]].assign(_disc="duplicate_posting"),
            pd.DataFrame(
                {"transaction_id": missing_tx, "ledger_id": "", "_disc": "missing_ledger_entry"}
            ),
        ],
        ignore_index=True,
    ).rename(columns={"_disc": "discrepancy_type"})
    labels = labels.sort_values("transaction_id", kind="stable").reset_index(drop=True)
    return ledger.drop(columns=["_disc", "_is_dup"]), labels
