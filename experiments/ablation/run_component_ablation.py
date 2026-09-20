"""EXP-ABL-01: which deterministic component contributes what (engine-floor ablation, RQ-system).

No language model is involved. Every configuration flags a case when the engine floor is REVIEW.
Headline set: the 24 held-out evaluation cases. Sensitivity set: all distinct un-injected cases
(train + val + eval; the anomaly model saw part of the training period, so this set flatters it).

    python experiments/ablation/run_component_ablation.py
"""

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any

from evaluation.ablation.components import Config, configs, evaluate, flagged
from llm.fine_tuning.dataset import CaseRecord, read_jsonl


def _flag(cfg: Config, record: CaseRecord) -> bool:
    return flagged(record, cfg)


def fmt(cell: dict[str, float]) -> str:
    if "lo" not in cell:
        return f"{cell['mean']:.2f} (n flagged={cell['n']})"
    return f"{cell['mean']:.2f} [{cell['lo']:.2f}, {cell['hi']:.2f}] n={cell['n']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/finetune"))
    ap.add_argument("--out", type=Path, default=Path("evaluation/reports/ablation"))
    args = ap.parse_args()

    eval_set = read_jsonl(args.data / "eval.jsonl")
    seen: set[str] = set()
    pooled: list[CaseRecord] = []
    for name in ("eval", "val", "train"):
        for r in read_jsonl(args.data / f"{name}.jsonl"):
            key = r.meta.get("transaction_id", r.case_id)
            if r.injected is None and key not in seen:
                seen.add(key)
                pooled.append(r)

    report: dict[str, Any] = {
        "note": "SYNTHETIC data. Engine-floor ablation, no LLM. 95% CIs: case bootstrap.",
        "sets": {"heldout_eval": len(eval_set), "pooled_uninjected": len(pooled)},
        "results": {},
    }
    lines = [
        "# EXP-ABL-01: engine-floor component ablation\n",
        "SYNTHETIC. Flag = engine floor is REVIEW.\n",
    ]
    for set_name, records in (("heldout_eval", eval_set), ("pooled_uninjected", pooled)):
        report["results"][set_name] = {}
        lines += [
            f"\n## {set_name} (n={len(records)})\n",
            "| configuration | recall: reconciliation | recall: behavioural | false-flag rate: clean | precision |",
            "|---|---|---|---|---|",
        ]
        for cfg in configs():
            res = evaluate(records, partial(_flag, cfg))
            report["results"][set_name][cfg.name] = res
            lines.append(
                f"| {cfg.name} | {fmt(res['recall_reconciliation'])} | {fmt(res['recall_behavioural'])} "
                f"| {fmt(res['false_flag_rate_clean'])} | {res['precision']['mean']:.2f} |"
            )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "component_ablation.json").write_text(json.dumps(report, indent=2))
    (args.out / "component_ablation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
