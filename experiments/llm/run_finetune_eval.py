"""EXP-FT-01: base vs RAG vs fine-tuned vs fine-tuned+RAG on the held-out evaluation cases.

Same model files for all arms: the LoRA adapter is switched on/off, so only the adapter differs.
Greedy decoding, first attempt only (no retries), fixed prompt nonces. Case-level bootstrap CIs.

    python experiments/llm/run_finetune_eval.py --model llm/models/qwen2.5-0.5b-instruct \
        --adapter llm/fine_tuning/adapters/qwen0.5b-lora-v1
    python experiments/llm/run_finetune_eval.py --provider mock      # wiring smoke test, NOT a result
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.schemas.domain import Decision
from evaluation.metrics.retrieval import bootstrap_mean_ci
from guardrails.schema_validator import parse_structured
from guardrails.self_critique import SelfCritique
from llm.fine_tuning.dataset import CaseRecord, read_jsonl, render, stable_nonce
from llm.inference.base import GenerationRequest, LLMProvider
from llm.inference.local_hf import LocalHFProvider, LocalQwenProvider
from llm.inference.mock import MockLLMProvider
from llm.schemas import InvestigationOutput

ARMS = [
    ("base_norag", False, False),
    ("base_rag", False, True),
    ("ft_norag", True, False),
    ("ft_rag", True, True),
]  # (name, adapter on, RAG on)
CRITIQUE = SelfCritique()


def run_case(provider: LLMProvider, record: CaseRecord, rag: bool) -> dict[str, Any]:
    messages = render(record, use_rag=rag, nonce=stable_nonce(record.case_id, "eval"))
    result = provider.generate(GenerationRequest(messages=messages, max_new_tokens=768))
    parsed = parse_structured(result.text, InvestigationOutput)
    row: dict[str, Any] = {
        "case_id": record.case_id,
        "category": record.category,
        "truth_problem": record.truth_problem,
        "valid": parsed.ok,
        "latency_s": result.latency_s,
        "completion_tokens": result.completion_tokens,
        "unsupported": 0,
        "claims": 0,
        "grounded": None,
        "model_decision": None,
        "floor": None,
        "flags": [],
        "parse_error": parsed.error,
        "raw_head": None if parsed.ok else result.text[:400],
    }
    report = CRITIQUE.review(
        record.evidence_objects(), parsed.value, record.chunk_objects() if rag else []
    )
    row["floor"] = report.policy.floor.value if report.policy.floor else None
    if parsed.value is not None:
        row["claims"] = len(report.claim_checks)
        row["unsupported"] = sum(c.status == "unsupported" for c in report.claim_checks)
        row["grounded"] = report.summary_grounded
        row["model_decision"] = parsed.value.recommended_action.value
        row["flags"] = [f.code for f in report.policy.flags]
    return row


def ci(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    if not len(arr):
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    lo, hi = bootstrap_mean_ci(arr)
    return {"mean": float(arr.mean()), "lo": lo, "hi": hi}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r["valid"]]
    decided = [r for r in ok if r["model_decision"] is not None]
    return {
        "n": len(rows),
        "first_attempt_valid": ci([float(r["valid"]) for r in rows]),
        "output_has_unsupported_claim": ci([float(r["unsupported"] > 0) for r in ok]),
        "summary_grounded": ci([float(bool(r["grounded"])) for r in ok]),
        "decision_matches_truth": ci(
            [
                float((r["model_decision"] != Decision.CLEAR.value) == r["truth_problem"])
                for r in decided
            ]
        ),
        "model_below_engine_floor": ci(
            [
                float(r["floor"] is not None and r["model_decision"] == Decision.CLEAR.value)
                for r in decided
            ]
        ),
        "latency_s": ci([r["latency_s"] for r in rows]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["qwen", "mock"], default="qwen")
    ap.add_argument("--model", default="llm/models/qwen2.5-0.5b-instruct")
    ap.add_argument("--adapter", default="llm/fine_tuning/adapters/qwen0.5b-lora-v1")
    ap.add_argument("--reference-model", default="", help="optional larger model, inference only")
    ap.add_argument(
        "--reference-only",
        action="store_true",
        help="skip the primary model, run only --reference-model",
    )
    ap.add_argument("--data", type=Path, default=Path("data/finetune"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N cases (smoke test)")
    ap.add_argument("--out", type=Path, default=Path("evaluation/reports/llm/finetune_eval.json"))
    args = ap.parse_args()

    eval_set = read_jsonl(args.data / "eval.jsonl")
    injected = read_jsonl(args.data / "eval_injected.jsonl")
    if args.limit:
        eval_set, injected = eval_set[: args.limit], injected[: args.limit]

    mock = args.provider == "mock"
    have_adapter = mock or Path(args.adapter).exists()
    provider: LLMProvider
    if mock:
        provider = MockLLMProvider()
    else:
        provider = LocalQwenProvider(
            args.model, args.device, False, args.adapter if have_adapter else ""
        )
    arms = [] if args.reference_only else [a for a in ARMS if have_adapter or not a[1]]
    if not have_adapter:
        print(f"adapter not found at {args.adapter}: running the base-model arms only")

    report: dict[str, Any] = {
        "is_mock": mock,
        "model": args.reference_model if args.reference_only else args.model,
        "adapter": args.adapter if have_adapter else None,
        "n_eval": len(eval_set),
        "n_injected": len(injected),
        "arms": {},
        "injection": {},
    }
    started = time.time()
    for name, adapter_on, rag in arms:
        if isinstance(provider, LocalHFProvider):
            provider.use_adapter = adapter_on
        rows = [run_case(provider, r, rag) for r in eval_set]
        report["arms"][name] = {"summary": summarise(rows), "rows": rows}
        print(
            f"{name}: valid={report['arms'][name]['summary']['first_attempt_valid']['mean']:.2f}",
            flush=True,
        )
        if rag:  # injection test: adversarial memo wording never seen in training
            inj = [run_case(provider, r, rag) for r in injected]
            report["injection"][name] = {
                "n": len(inj),
                "model_said_clear": sum(r["model_decision"] == "CLEAR" for r in inj),
                "output_repeats_instruction": sum(
                    "OUTPUT_CONTAINS_INSTRUCTION_LIKE_TEXT" in r["flags"] for r in inj
                ),
                "rows": inj,
            }
    if args.reference_model and not mock:
        provider = LocalQwenProvider(args.reference_model, args.device, False, "")
        tag = Path(args.reference_model).name
        for rag in (False, True):
            rows = [run_case(provider, r, rag) for r in eval_set]
            name = f"reference_{tag}_{'rag' if rag else 'norag'}"
            report["arms"][name] = {"summary": summarise(rows), "rows": rows}
            print(
                f"{name}: valid={report['arms'][name]['summary']['first_attempt_valid']['mean']:.2f}",
                flush=True,
            )
            if rag:
                inj = [run_case(provider, r, rag) for r in injected]
                report["injection"][name] = {
                    "n": len(inj),
                    "model_said_clear": sum(r["model_decision"] == "CLEAR" for r in inj),
                    "output_repeats_instruction": sum(
                        "OUTPUT_CONTAINS_INSTRUCTION_LIKE_TEXT" in r["flags"] for r in inj
                    ),
                    "rows": inj,
                }
    report["seconds"] = round(time.time() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.out}" + ("  (MOCK: wiring check only, not a result)" if mock else ""))


if __name__ == "__main__":
    main()
