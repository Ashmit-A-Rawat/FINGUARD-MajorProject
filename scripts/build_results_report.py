"""Assemble docs/research/results-summary.md from the saved JSON reports (nothing is typed by hand).

A research question whose report is missing is listed as PENDING, never filled in.

    python scripts/build_results_report.py
"""

import json
from pathlib import Path
from typing import Any

from evaluation.ablation.guardrail_layers import layer_table

REPORTS = Path("evaluation/reports")
OUT = Path("docs/research/results-summary.md")


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def cell(c: dict[str, float] | None) -> str:
    if not c or c.get("mean") != c.get("mean"):  # missing or NaN
        return "n/a"
    if "lo" in c:
        return f"{c['mean']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}]"
    return f"{c['mean']:.2f}"


def ci_pair(pair: list[float]) -> str:
    return f"[{pair[0]:.3f}, {pair[1]:.3f}]"


def rq1() -> list[str]:
    lines = ["| seed | system | precision | recall | F1 (95% CI) |", "|---|---|---|---|---|"]
    found = False
    for path in sorted((REPORTS / "kyc").glob("kyc_benchmark_*.json")):
        d = json.loads(path.read_text())
        for name in ("fuzzy", "dense", "hybrid", "full"):
            s = d["systems"].get(name)
            if s:
                found = True
                lines.append(
                    f"| {d['environment']['seed']} | {name} | {s['test']['precision']:.3f} | "
                    f"{s['test']['recall']:.3f} | {s['test']['f1']:.3f} {ci_pair(s['test_ci95']['f1'])} |"
                )
    return lines if found else ["PENDING: run `make kyc-benchmark`."]


def rq2() -> list[str]:
    lines = ["| seed | model | family | PR-AUC (95% CI, customer-clustered) |", "|---|---|---|---|"]
    found = False
    for path in sorted((REPORTS / "anomaly").glob("anomaly_benchmark_*.json")):
        d = json.loads(path.read_text())
        for name, m in d["models"].items():
            found = True
            lines.append(
                f"| {d['environment']['seed']} | {name} | {m['family']} | "
                f"{m['test']['pr_auc']:.3f} {ci_pair(m['test_ci95']['pr_auc'])} |"
            )
    return lines if found else ["PENDING: run `make anomaly-benchmark`."]


def llm_arms() -> tuple[dict[str, Any], list[str]]:
    """Merge every finetune_eval*.json (laptop base arms + PC fine-tuned arms)."""
    arms: dict[str, Any] = {}
    sources: list[str] = []
    for path in sorted((REPORTS / "llm").glob("finetune_eval*.json")):
        d = json.loads(path.read_text())
        if d.get("is_mock"):
            continue  # a mock run is a wiring check, never a result
        sources.append(f"{path.name} ({d['model']})")
        for name, arm in d["arms"].items():
            arms[name] = arm
        for name, inj in d.get("injection", {}).items():
            arms.setdefault(name, {})["injection"] = inj
    return arms, sources


def rq3_rq4_rq6(arms: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    head = (
        "| arm | first-attempt valid | outputs with an unsupported claim | summary grounded | "
        "decision matches truth | model CLEAR below engine floor | latency s |"
    )
    sep = "|---|---|---|---|---|---|---|"
    if not arms:
        pend = ["PENDING: run `experiments/llm/run_finetune_eval.py`."]
        return pend, pend, pend
    rows = [head, sep]
    for name, arm in arms.items():
        s = arm.get("summary")
        if s:
            rows.append(
                f"| {name} | {cell(s['first_attempt_valid'])} | {cell(s['output_has_unsupported_claim'])} | "
                f"{cell(s['summary_grounded'])} | {cell(s['decision_matches_truth'])} | "
                f"{cell(s['model_below_engine_floor'])} | {cell(s['latency_s'])} |"
            )
    layers = [
        "| arm | valid outputs | layer | problem cases left CLEAR | clean cases raised to REVIEW |",
        "|---|---|---|---|---|",
    ]
    for name, arm in arms.items():
        if "rows" not in arm:
            continue
        t = layer_table(arm["rows"])
        for layer, label in (
            ("model", "model only"),
            ("floor", "+ engine floor"),
            ("full", "+ unsupported-claim rule"),
        ):
            layers.append(
                f"| {name} | {t['n_valid_outputs']}/{t['n_rows']} | {label} | "
                f"{cell(t[layer]['problem_cases_left_CLEAR'])} | {cell(t[layer]['clean_cases_raised_to_REVIEW'])} |"
            )
    have_ft = any(n.startswith("ft_") for n in arms)
    rq6 = (
        rows
        if have_ft
        else [
            "PENDING: the fine-tuned arms need the adapter trained on a GPU machine "
            "(docs/architecture/finetune-on-gpu-pc.md). Base-model arms above are the baseline."
        ]
    )
    return rows, layers, rq6


def rq5() -> list[str]:
    d = load(REPORTS / "agents" / "orchestration_ablation_qwen.json")
    if not d:
        return ["PENDING: run `experiments/agents/run_orchestration_ablation.py`."]
    a, p = d["decision_accuracy"], d["paired_multi_vs_single_model_only"]
    return [
        f"Model `{d['model']}`, n={d['n_cases']} held-out cases, greedy decoding, one run.",
        "",
        "| variant | decision agrees with label | first-attempt valid | wall s per case |",
        "|---|---|---|---|",
        f"| multi-agent, model decision | {cell(a['multi_model_only'])} | {cell(d['first_attempt_valid']['multi'])} | {cell(d['wall_s']['multi'])} |",
        f"| multi-agent, after guardrails | {cell(a['multi_after_guardrails'])} | | |",
        f"| single-agent baseline | {cell(a['single_model_only'])} | {cell(d['first_attempt_valid']['single'])} | {cell(d['wall_s']['single'])} |",
        "",
        f"Paired: multi right & single wrong on {p['multi_right_single_wrong']} cases; single right & multi wrong on {p['single_right_multi_wrong']}.",
    ]


def component_and_adversarial() -> list[str]:
    out: list[str] = []
    abl = REPORTS / "ablation" / "component_ablation.md"
    adv = REPORTS / "adversarial" / "memo_injection_sweep.md"
    out.append(
        f"- Engine-floor component ablation: `{abl}` "
        + ("(present)" if abl.exists() else "(PENDING)")
    )
    out.append(f"- Memo-injection sweep: `{adv}` " + ("(present)" if adv.exists() else "(PENDING)"))
    t = load(REPORTS / "adversarial" / "tripwire_eval_final.json")
    if t:
        v4 = t["sets"]["v4"]["detectors"]
        out.append(
            "- Tripwire on the blind set v4: "
            + "; ".join(
                f"{n} recall {d['caught']}/{t['sets']['v4']['attacks']}, false positives {d['false_positives']}/{t['sets']['v4']['benign']}"
                for n, d in v4.items()
            )
        )
    a = load(REPORTS / "adversarial" / "memo_injection_sweep.json")
    if a:
        c = a["classes"]
        lowered = sum(v["floor_lowered"] for v in c.values())
        out.append(
            f"- Across {len(c)} wording classes the floor of an engine-flagged problem case was lowered {lowered} times; "
            f"the improved tripwire missed {len(a['attack_wordings_never_caught'])} of the v1 development wordings it was tuned on (see the blind-set line above for the honest estimate)."
        )
    return out


def main() -> None:
    arms, sources = llm_arms()
    t3, t4, t6 = rq3_rq4_rq6(arms)
    parts = [
        "# Results summary (generated)\n",
        "Generated by `scripts/build_results_report.py` from `evaluation/reports/`. All data is SYNTHETIC. "
        "Read each experiment's threats to validity in `experiments.md` before quoting a number.\n",
        "## RQ1: hybrid entity resolution (KYC)\n",
        *rq1(),
        "",
        "## RQ2: supervised vs unsupervised vs neural anomaly detection\n",
        *rq2(),
        "",
        "## RQ3: does RAG improve grounding? (LLM arms, held-out cases)\n",
        f"Sources: {', '.join(sources) if sources else 'none'}\n",
        *t3,
        "",
        "## RQ4: what do the guardrail layers add?\n",
        *t4,
        "",
        "## RQ5: multi-agent orchestration vs single agent\n",
        *rq5(),
        "",
        "## RQ6: fine-tuning vs prompting vs RAG\n",
        *t6,
        "",
        "## Component ablation and adversarial tests\n",
        *component_and_adversarial(),
        "",
    ]
    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
