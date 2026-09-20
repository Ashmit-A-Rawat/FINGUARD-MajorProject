"""RQ4 at scale: what each guardrail layer adds on top of the raw model decision.

Layers, applied cumulatively to rows produced by experiments/llm/run_finetune_eval.py:
  model only            -> the model's own recommended_action
  + engine floor        -> max(model, floor)              (policy_checks.engine_floor)
  + unsupported claims  -> REVIEW if any claim is unsupported  (the full SelfCritique decision)
Outputs that failed to parse have no model decision; they are counted separately and excluded.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from evaluation.metrics.retrieval import bootstrap_mean_ci

RANK = {"CLEAR": 0, "REVIEW": 1, "ESCALATE": 2}


def _decision(row: dict[str, Any], layer: str) -> str:
    d: str = row["model_decision"]
    if layer in ("floor", "full") and row["floor"] is not None and RANK[row["floor"]] > RANK[d]:
        d = row["floor"]
    if layer == "full" and row["unsupported"] > 0 and RANK[d] < RANK["REVIEW"]:
        d = "REVIEW"
    return d


def _ci(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    if not len(arr):
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    lo, hi = bootstrap_mean_ci(arr)
    return {"mean": float(arr.mean()), "lo": lo, "hi": hi, "n": len(arr)}


def layer_table(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r["model_decision"] is not None]
    out: dict[str, Any] = {"n_rows": len(rows), "n_valid_outputs": len(valid)}
    for layer in ("model", "floor", "full"):
        out[layer] = {
            "problem_cases_left_CLEAR": _ci(
                [float(_decision(r, layer) == "CLEAR") for r in valid if r["truth_problem"]]
            ),
            "clean_cases_raised_to_REVIEW": _ci(
                [float(_decision(r, layer) != "CLEAR") for r in valid if not r["truth_problem"]]
            ),
        }
    return out
