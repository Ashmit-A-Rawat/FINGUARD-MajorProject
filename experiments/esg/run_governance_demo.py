"""Demonstration (NOT a validated benchmark) of the governance-risk indicator on real synthetic
transaction history. Deterministic, no language model involved.

There is no ground-truth label for "governance risk" in the generator (ESG was not part of the
original synthetic-data design), so this reports descriptive counts only -- how often each signal
fires over real data -- not precision/recall or a confidence interval. Treat it as evidence the
code runs correctly on real data, not as a performance claim.

    python experiments/esg/run_governance_demo.py --data-dir data/synthetic/small
"""

import argparse
import json
from pathlib import Path
from typing import Any

from data_pipeline.pipeline import run_pipeline
from esg.governance_scoring import GovernanceConfig, governance_evidence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/synthetic/small"))
    ap.add_argument("--out", type=Path, default=Path("evaluation/reports/esg"))
    args = ap.parse_args()

    result = run_pipeline(args.data_dir)
    if result.quarantine:
        raise RuntimeError(
            f"{len(result.quarantine)} rows quarantined; refusing to run on bad data"
        )
    store = result.store
    config = GovernanceConfig()

    rows: list[dict[str, Any]] = []
    for customer_id, txs in store.transactions_by_customer.items():
        ordered = sorted(txs, key=lambda t: t.timestamp)
        for i, tx in enumerate(ordered):
            history = ordered[:i]
            if len(history) < 3:  # too little history for a meaningful governance check
                continue
            ev = governance_evidence(tx, history, tx.timestamp, config)
            if ev.payload["flagged"]:
                rows.append(
                    {
                        "customer_id": customer_id,
                        "evidence_id": ev.evidence_id,
                        "description": ev.description,
                        "payload": ev.payload,
                    }
                )

    total_checked = sum(
        max(0, len(sorted(txs, key=lambda t: t.timestamp)) - 3)
        for txs in store.transactions_by_customer.values()
    )
    new_border = sum(r["payload"]["new_cross_border_counterparty"] for r in rows)
    structuring = sum(r["payload"]["structuring_suspected"] for r in rows)
    report = {
        "notice": "SYNTHETIC data. Descriptive demo of the governance scorer, not a validated benchmark "
        "(no ground-truth governance-risk labels exist in the generator).",
        "config": config.model_dump(),
        "transactions_checked": total_checked,
        "flagged": len(rows),
        "flagged_new_cross_border_counterparty": new_border,
        "flagged_structuring_suspected": structuring,
        "examples": rows[:15],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "governance_demo.json").write_text(json.dumps(report, indent=2, default=str))
    print(
        f"checked {total_checked} transactions; flagged {len(rows)} "
        f"({new_border} new cross-border counterparty, {structuring} possible structuring)"
    )
    for r in rows[:5]:
        print(" -", r["evidence_id"], r["description"])


if __name__ == "__main__":
    main()
