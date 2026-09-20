"""EXP-ADV-01: payment-memo injection sweep against the deterministic layers (no LLM).

For every wording class in evaluation/adversarial_dataset/memo_injections.json, the wording is written
into the memo of each real evaluation case and the engine floor is recomputed. Questions:
  1. TRIPWIRE: how often is instruction-like text flagged, per class (recall), and how often is a
     BENIGN memo flagged (false positives)?
  2. PROTECTION: does an injection ever LOWER the floor of a case the engines flagged? (It must not.)
  3. RESIDUAL RISK: problem cases the engines never flagged (floor None) rely on the model alone.

    python experiments/adversarial/run_adversarial_suite.py
"""

import argparse
import json
from pathlib import Path
from typing import Any

from backend.app.schemas.domain import Evidence
from evaluation.ablation.components import ALL_PREFIXES
from guardrails.policy_checks import PolicyConfig, engine_floor
from llm.fine_tuning.dataset import CaseRecord, read_jsonl

CONFIG = PolicyConfig()


def with_memo(record: CaseRecord, memo: str) -> list[Evidence]:
    out = []
    for e in record.evidence_objects():
        if e.source.value == "transaction":
            e = e.model_copy(update={"payload": {**e.payload, "payment_memo": memo}})
        out.append(e)
    return out


def tripped(reasons: list[str]) -> bool:
    return any("instruction-like text" in r for r in reasons)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/finetune"))
    ap.add_argument(
        "--wordings", type=Path, default=Path("evaluation/adversarial_dataset/memo_injections.json")
    )
    ap.add_argument("--out", type=Path, default=Path("evaluation/reports/adversarial"))
    args = ap.parse_args()

    classes: dict[str, list[str]] = json.loads(args.wordings.read_text())["classes"]
    records = [r for r in read_jsonl(args.data / "eval.jsonl") if r.injected is None]
    flagged_problem = [
        r for r in records if r.truth_problem and engine_floor(r.evidence_objects(), CONFIG)[0]
    ]
    unflagged_problem = [r for r in records if r.truth_problem and r not in flagged_problem]
    clean = [r for r in records if not r.truth_problem]

    report: dict[str, Any] = {
        "note": "SYNTHETIC. Deterministic layers only; wordings hand-written by the author.",
        "cases": {
            "problem_flagged_by_engines": len(flagged_problem),
            "problem_missed_by_engines": len(unflagged_problem),
            "clean": len(clean),
        },
        "classes": {},
    }
    lines = [
        "# EXP-ADV-01: memo injection sweep\n",
        f"Cases: {len(flagged_problem)} problem cases flagged by the engines, "
        f"{len(unflagged_problem)} problem cases the engines missed, {len(clean)} clean.\n",
        "| wording class | wordings | tripwire fires (per wording x case) | floor lowered on flagged problem cases |",
        "|---|---|---|---|",
    ]
    for cls, wordings in classes.items():
        trips = lowered = total = 0
        per_wording = {}
        for w in wordings:
            hit = 0
            for r in records:
                floor, reasons = engine_floor(with_memo(r, w), CONFIG)
                total += 1
                hit += tripped(reasons)
                trips += tripped(reasons)
            for r in flagged_problem:
                floor, _ = engine_floor(with_memo(r, w), CONFIG)
                lowered += floor is None
            per_wording[w] = hit / len(records)
        report["classes"][cls] = {
            "tripwire_rate": trips / total,
            "floor_lowered": lowered,
            "per_wording_tripwire_rate": per_wording,
        }
        lines.append(
            f"| {cls} | {len(wordings)} | {trips}/{total} = {trips / total:.2f} | {lowered} |"
        )
    # which wordings evade (fire on none of the cases)
    evaders = [
        w
        for c in classes
        if not c.startswith("benign") and not c.startswith("hard_benign")
        for w, rate in report["classes"][c]["per_wording_tripwire_rate"].items()
        if rate == 0
    ]
    report["attack_wordings_never_caught"] = evaders
    lines += [
        "",
        f"Attack wordings the tripwire never caught: {len(evaders)}",
        *[f"- {w}" for w in evaders],
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "memo_injection_sweep.json").write_text(json.dumps(report, indent=2))
    (args.out / "memo_injection_sweep.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    _ = ALL_PREFIXES


if __name__ == "__main__":
    main()
