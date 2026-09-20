"""Run the multi-agent workflow and the single-agent baseline on the same held-out cases.

    python experiments/agents/run_workflow_demo.py --provider qwen --model llm/models/qwen2.5-1.5b-instruct

Cases come from AFTER the anomaly model's training period (no optimistic scores): 2 reconciliation
problems, 2 behavioural anomalies, 2 clean transactions. This is a functional demonstration with a
latency breakdown (groundwork for RQ5), NOT an accuracy evaluation: n is 6 and nothing is validated
until Phase 10.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.baseline import SingleAgentBaseline
from agents.context import build_context
from agents.coordinator.workflow import CaseWorkflow
from backend.app.core.config import Settings
from knowledge_base.embeddings.embedder import SentenceTransformerEmbedder
from llm.inference.factory import create_provider


def pick_cases(ctx: Any, data_dir: Path, seed: int) -> list[tuple[str, str, str]]:
    """(category, customer_id, transaction_id), all after the anomaly model's training period."""
    rng = np.random.default_rng(seed)
    txs = {t.transaction_id: t for v in ctx.store.transactions_by_customer.values() for t in v}
    late = {tid for tid, t in txs.items() if pd.Timestamp(t.timestamp) >= ctx.anomaly.trained_until}
    ledger = pd.read_csv(data_dir / "ledger_labels.csv", keep_default_na=False)
    labels = pd.read_csv(data_dir / "transaction_labels.csv", keep_default_na=False)
    recon = sorted(set(ledger["transaction_id"]) & late)
    behav = sorted(
        set(labels[labels["is_anomaly"].astype(str).eq("True")]["transaction_id"]) & late
    )
    problem = set(ledger["transaction_id"]) | set(
        labels[labels["is_anomaly"].astype(str).eq("True")]["transaction_id"]
    )
    clean = sorted(late - problem)
    chosen: list[tuple[str, str, str]] = []
    for category, pool in (("reconciliation", recon), ("behavioural", behav), ("clean", clean)):
        for i in rng.permutation(len(pool))[:2]:
            tid = pool[int(i)]
            chosen.append((category, txs[tid].customer_id, tid))
    return chosen


def run(provider_name: str, model: str, data_dir: Path, seed: int, out_dir: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    provider = create_provider(Settings(llm_provider=provider_name, llm_model=model))
    ctx = build_context(data_dir, SentenceTransformerEmbedder(), provider)
    setup_s = time.perf_counter() - t0
    workflow, baseline = CaseWorkflow(ctx), SingleAgentBaseline(ctx)
    rows: list[dict[str, Any]] = []
    for n, (category, customer, tid) in enumerate(pick_cases(ctx, data_dir, seed), start=1):
        started = time.perf_counter()
        state = workflow.run(workflow.create_case(customer, [tid]))
        multi_s = time.perf_counter() - started
        base = baseline.run(customer, [tid])
        steps = {
            e.action: e.duration_ms / 1000 for e in state.audit_trail if e.action != "create_case"
        }
        rows.append(
            {
                "case": f"{category}-{tid}",
                "category": category,
                "multi_agent": {
                    "status": state.status.value,
                    "failed_step": state.failed_step,
                    "proposed_decision": state.report.proposed_decision.value
                    if state.report and state.report.proposed_decision
                    else None,
                    "warnings": len(state.report.warnings) if state.report else None,
                    "advisory_decision": state.report.advisory_decision.value
                    if state.report and state.report.advisory_decision
                    else None,
                    "claim_summary": state.report.claim_summary if state.report else None,
                    "validated": state.report.validated if state.report else None,
                    "policy_flags": state.report.policy_flags if state.report else None,
                    "engine_facts": len(state.report.engine_facts) if state.report else None,
                    "attempts": state.investigation_meta.attempts
                    if state.investigation_meta
                    else None,
                    "wall_s": multi_s,
                    "step_seconds": steps,
                },
                "single_agent": {
                    "proposed_decision": base.investigation.recommended_action.value
                    if base.investigation
                    else None,
                    "valid": base.investigation is not None,
                    "attempts": base.meta.attempts,
                    "wall_s": base.wall_seconds,
                },
            }
        )
        print(
            f"[{'#' * round(30 * n / 6):-<30}] {n}/6 {category:15s} multi={multi_s:5.1f}s "
            f"single={base.wall_seconds:5.1f}s",
            flush=True,
        )

    step_names = list(rows[0]["multi_agent"]["step_seconds"])
    mean_steps = {
        s: float(np.mean([r["multi_agent"]["step_seconds"][s] for r in rows])) for s in step_names
    }
    total = sum(mean_steps.values())
    report = {
        "notice": "SYNTHETIC; functional demo + latency breakdown, not an accuracy evaluation (n=6).",
        "provider": provider.name,
        "model": provider.model,
        "is_mock": provider.is_mock,
        "setup_seconds_once": setup_s,
        "n_cases": len(rows),
        "mean_step_seconds_multi_agent": mean_steps,
        "llm_share_of_multi_agent_latency": mean_steps["investigate"] / total,
        "mean_wall_s": {
            "multi_agent": float(np.mean([r["multi_agent"]["wall_s"] for r in rows])),
            "single_agent": float(np.mean([r["single_agent"]["wall_s"] for r in rows])),
        },
        "all_reached_human_review": all(r["multi_agent"]["status"] == "human_review" for r in rows),
        "cases": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"workflow_demo_{provider.name}.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--model", default="")
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic/small"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/reports/agents"))
    args = parser.parse_args()
    report = run(args.provider, args.model, args.data_dir, args.seed, args.output_dir)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
