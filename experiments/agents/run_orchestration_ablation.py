"""EXP-ORCH-01 (RQ5): multi-agent workflow vs single-agent baseline on the held-out cases.

Same engines, same evidence builders, same model; what differs is orchestration + targeted retrieval
(see agents/baseline.py). Reports first-attempt validity, decision agreement with the generator's labels
(problem = anything but CLEAR), model-only vs post-guardrail decisions, and latency, with case-bootstrap
CIs and a paired comparison. Greedy decoding, so a repeat run reproduces the outputs; latency varies.

    python experiments/agents/run_orchestration_ablation.py --provider qwen --model llm/models/qwen2.5-0.5b-instruct
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from agents.baseline import SingleAgentBaseline
from agents.context import build_context
from agents.coordinator.workflow import CaseWorkflow
from backend.app.core.config import Settings
from evaluation.metrics.retrieval import bootstrap_mean_ci
from knowledge_base.embeddings.embedder import SentenceTransformerEmbedder
from llm.fine_tuning.dataset import read_jsonl
from llm.inference.factory import create_provider


def ci(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    lo, hi = bootstrap_mean_ci(arr) if len(arr) else (float("nan"), float("nan"))
    return {"mean": float(arr.mean()) if len(arr) else float("nan"), "lo": lo, "hi": hi}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="qwen")
    ap.add_argument("--model", default="llm/models/qwen2.5-0.5b-instruct")
    ap.add_argument("--data-dir", type=Path, default=Path("data/synthetic/small"))
    ap.add_argument("--cases", type=Path, default=Path("data/finetune/eval.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("evaluation/reports/agents"))
    args = ap.parse_args()

    records = read_jsonl(args.cases)[: args.limit or None]
    started = time.perf_counter()
    provider = create_provider(Settings(llm_provider=args.provider, llm_model=args.model))
    ctx = build_context(args.data_dir, SentenceTransformerEmbedder(), provider)
    workflow, baseline = CaseWorkflow(ctx), SingleAgentBaseline(ctx)
    setup_s = time.perf_counter() - started

    rows: list[dict[str, Any]] = []
    for n, r in enumerate(records, start=1):
        cust, tid = r.meta["customer_id"], r.meta["transaction_id"]
        t0 = time.perf_counter()
        state = workflow.run(workflow.create_case(cust, [tid]))
        multi_s = time.perf_counter() - t0
        base = baseline.run(cust, [tid])
        rep = state.report
        multi_model = rep.proposed_decision.value if rep and rep.proposed_decision else None
        multi_final = rep.advisory_decision.value if rep and rep.advisory_decision else None
        base_model = base.investigation.recommended_action.value if base.investigation else None
        rows.append(
            {
                "case_id": r.case_id,
                "category": r.category,
                "truth_problem": r.truth_problem,
                "multi": {
                    "valid_first_attempt": bool(
                        state.investigation_meta and state.investigation_meta.attempts == 1
                    ),
                    "model_decision": multi_model,
                    "final_decision": multi_final,
                    "wall_s": multi_s,
                    "status": state.status.value,
                },
                "single": {
                    "valid_first_attempt": base.meta.attempts == 1
                    and base.investigation is not None,
                    "model_decision": base_model,
                    "wall_s": base.wall_seconds,
                },
            }
        )
        print(
            f"{n}/{len(records)} {r.category:15s} multi={multi_s:5.1f}s single={base.wall_seconds:5.1f}s",
            flush=True,
        )

    def correct(decision: str | None, truth: bool) -> float:
        return float(decision is not None and (decision != "CLEAR") == truth)

    arms = {
        "multi_model_only": [
            correct(x["multi"]["model_decision"], x["truth_problem"]) for x in rows
        ],
        "multi_after_guardrails": [
            correct(x["multi"]["final_decision"], x["truth_problem"]) for x in rows
        ],
        "single_model_only": [
            correct(x["single"]["model_decision"], x["truth_problem"]) for x in rows
        ],
    }
    paired = [
        (a, b) for a, b in zip(arms["multi_model_only"], arms["single_model_only"], strict=True)
    ]
    report = {
        "notice": "SYNTHETIC. n cases held out in time; greedy decoding; one run per case.",
        "provider": provider.name,
        "model": provider.model,
        "is_mock": provider.is_mock,
        "setup_seconds_once": setup_s,
        "n_cases": len(rows),
        "decision_accuracy": {k: ci(v) for k, v in arms.items()},
        "paired_multi_vs_single_model_only": {
            "multi_right_single_wrong": sum(a == 1 and b == 0 for a, b in paired),
            "single_right_multi_wrong": sum(a == 0 and b == 1 for a, b in paired),
        },
        "first_attempt_valid": {
            "multi": ci([float(x["multi"]["valid_first_attempt"]) for x in rows]),
            "single": ci([float(x["single"]["valid_first_attempt"]) for x in rows]),
        },
        "wall_s": {
            "multi": ci([x["multi"]["wall_s"] for x in rows]),
            "single": ci([x["single"]["wall_s"] for x in rows]),
        },
        "cases": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"orchestration_ablation_{provider.name}.json"
    dest.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=1, default=str))


if __name__ == "__main__":
    main()
