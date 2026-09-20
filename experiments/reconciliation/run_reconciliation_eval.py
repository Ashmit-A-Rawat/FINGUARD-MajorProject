"""Evaluate the reconciliation rules against the synthetic ledger ground truth.

    python experiments/reconciliation/run_reconciliation_eval.py --preset medium

Ground truth (ledger_labels.csv) covers six injected discrepancy types. Rules REC-001..REC-008
map onto them (REC-006 and REC-007 have no injected counterpart, so any hit is a false positive).
Status rules (REC-009/010) flag failed / stale-pending settlements: those statuses occur
in the generator's NORMAL data and are not labelled discrepancies, so they are reported
separately rather than counted as errors.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from data_pipeline.pipeline import run_pipeline
from reconciliation.config import ReconciliationConfig
from reconciliation.service import ReconciliationService

RULE_TO_LABEL = {
    "REC-001": "missing_ledger_entry",
    "REC-002": "duplicate_posting",
    "REC-003": "contradictory_amount",
    "REC-004": "currency_mismatch",
    "REC-005": "missing_reference",
    "REC-008": "late_posting",
}
CORE_RULES = {f"REC-00{i}" for i in range(1, 9)}


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
    }


def run(data_dir: Path, out_dir: Path, preset: str) -> dict[str, Any]:
    result = run_pipeline(data_dir)
    store = result.store
    transactions = [t for txs in store.transactions_by_customer.values() for t in txs]
    as_of = max(t.timestamp for t in transactions)
    service = ReconciliationService(ReconciliationConfig())
    results = service.reconcile_many(transactions, store.ledger_by_transaction, as_of)

    labels = pd.read_csv(data_dir / "ledger_labels.csv", keep_default_na=False)
    truth: dict[str, str] = dict(
        zip(labels["transaction_id"], labels["discrepancy_type"], strict=True)
    )

    predicted: dict[str, set[str]] = {}  # transaction -> label types implied by core rules
    fired: Counter[str] = Counter()
    for r in results:
        fired.update(d.rule_id for d in r.discrepancies)
        core = {d.rule_id for d in r.discrepancies if d.rule_id in CORE_RULES}
        if core:
            predicted[r.transaction_id] = {
                RULE_TO_LABEL.get(rid, f"unlabelled:{rid}") for rid in core
            }

    per_type: dict[str, Any] = {}
    for label in RULE_TO_LABEL.values():
        actual = {t for t, kind in truth.items() if kind == label}
        flagged = {t for t, kinds in predicted.items() if label in kinds}
        per_type[label] = _prf(len(actual & flagged), len(flagged - actual), len(actual - flagged))

    any_truth, any_pred = set(truth), set(predicted)
    overall = _prf(len(any_truth & any_pred), len(any_pred - any_truth), len(any_truth - any_pred))
    misattributed = sum(1 for t in any_truth & any_pred if truth[t] not in predicted[t])
    false_pos_unlabelled = Counter(
        rid
        for r in results
        if r.transaction_id not in truth
        for rid in (d.rule_id for d in r.discrepancies)
        if rid in CORE_RULES
    )

    report = {
        "notice": "SYNTHETIC data; results do not represent real-bank performance.",
        "preset": preset,
        "config": json.loads(ReconciliationConfig().model_dump_json()),
        "counts": {
            "transactions": len(transactions),
            "reconciled": sum(r.status == "reconciled" for r in results),
            "with_discrepancy": sum(r.status == "discrepancy" for r in results),
            "ground_truth_discrepancies": len(truth),
        },
        "rules_fired": dict(sorted(fired.items())),
        "core_rules_transaction_level": overall,
        "misattributed_type": misattributed,
        "per_type": per_type,
        "core_rule_false_positives_by_rule": dict(false_pos_unlabelled),
        "status_rules_note": "REC-009/010 findings occur on unlabelled normal data (see docstring)",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"reconciliation_eval_{preset}.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation/reports/reconciliation")
    )
    args = parser.parse_args()
    report = run(
        args.data_dir or Path("data/synthetic") / args.preset, args.output_dir, args.preset
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
