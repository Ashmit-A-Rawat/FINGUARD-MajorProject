"""Evaluate the guardrails (evidence validator + policy floor).

    python experiments/guardrails/run_guardrail_eval.py

PART A: claim-perturbation benchmark. Claims are derived from real case evidence by templates (so
they ARE supported), then corrupted in known ways. Ground truth is by construction. This measures
whether the validator implementation does what it says. The templates and perturbations were
written by the same author as the validator, so it is a correctness check, not a measure of how it
performs on free-form model text.

PART B: the guardrails applied to the REAL Qwen outputs saved by EXP-LLM-01 (evidence rebuilt
deterministically): how many claims are supported / unsupported / unverifiable, and how the advisory
decision changes, including the paired prompt-injection cases.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from backend.app.schemas.domain import Evidence
from experiments.llm.run_structured_output_check import INJECTION, World, build_cases, make_case
from guardrails.evidence_validator import EvidenceIndex, check_claim
from guardrails.schema_validator import parse_structured
from guardrails.self_critique import SelfCritique
from knowledge_base.embeddings.embedder import SentenceTransformerEmbedder
from knowledge_base.service import KnowledgeBase, SourceSpec
from llm.schemas import InvestigationOutput

DECISION_RANK = {"CLEAR": 0, "REVIEW": 1, "ESCALATE": 2}


def derive_claims(case_evidence: list[Evidence]) -> list[dict[str, Any]]:
    """Supported claims, each (text, cited ids, kind, perturbation-relevant metadata)."""
    claims: list[dict[str, Any]] = []
    other_id = {e.evidence_id.split("-")[0]: e.evidence_id for e in case_evidence}
    for e in case_evidence:
        p = e.payload
        if e.evidence_id.startswith("TXN-"):
            claims.append(
                {
                    "text": f"The transaction {e.evidence_id} is a {p['type']} of "
                    f"{p['amount']:.2f} {p['currency']}",
                    "cites": [e.evidence_id],
                    "tag": "txn",
                }
            )
            claims.append(
                {
                    "text": f"The transaction occurred on {str(p['timestamp'])[:10]}",
                    "cites": [e.evidence_id],
                    "tag": "date",
                }
            )
        elif e.evidence_id.startswith("BEH-"):
            if p.get("prior_transactions"):
                claims.append(
                    {
                        "text": f"The customer had {p['prior_transactions']} earlier transactions",
                        "cites": [e.evidence_id],
                        "tag": "count",
                    }
                )
            if p.get("typical_prior_amount_ratio"):
                claims.append(
                    {
                        "text": f"The amount is {p['typical_prior_amount_ratio']} times the typical "
                        "earlier amount",
                        "cites": [e.evidence_id],
                        "tag": "ratio",
                    }
                )
        elif (
            e.evidence_id.startswith("REC-")
            and "rule_id" in p
            and p.get("expected")
            and p.get("actual")
        ):
            claims.append(
                {
                    "text": f"Rule {p['rule_id']} reports expected {p['expected']} but actual {p['actual']}",
                    "cites": [e.evidence_id],
                    "tag": "rec",
                }
            )
        elif e.evidence_id.startswith("KYC-"):
            claims.append(
                {
                    "text": f"The account is {p['account_age_days']} days old",
                    "cites": [e.evidence_id],
                    "tag": "kyc",
                }
            )
    for c in claims:
        c["wrong_cite"] = [
            next((i for pre, i in other_id.items() if pre != c["cites"][0].split("-")[0]), "")
        ]
    return claims


def perturb(claim: dict[str, Any]) -> list[tuple[str, str, list[str]]]:
    """(perturbation, text, cites) triples that are unsupported by construction."""
    text, cites, out = claim["text"], claim["cites"], []
    numbers = re.findall(
        r"(?<![\w.-])\d+(?:\.\d+)?(?![\w-])", re.sub(r"[A-Z]+-[\w-]+|\d{4}-\d{2}-\d{2}", " ", text)
    )
    if numbers and claim["tag"] in ("txn", "count", "ratio", "rec", "kyc"):
        n = numbers[-1] if claim["tag"] != "txn" else numbers[0]
        changed = f"{float(n) * 1.07 + 1:.{len(n.split('.')[1]) if '.' in n else 0}f}"
        out.append(("number_changed", text.replace(n, changed, 1), cites))
    if claim["tag"] == "txn":
        out.append(("id_invented", re.sub(r"TXN-\d+", "TXN-99999999", text), cites))
        out.append(
            (
                "currency_swapped",
                re.sub(
                    r"\b(USD|EUR|GBP|INR|AED|SGD|CAD|AUD)\b",
                    lambda m: "EUR" if m.group(1) != "EUR" else "USD",
                    text,
                ),
                cites,
            )
        )
    if claim["tag"] == "rec":
        out.append(("rule_swapped", re.sub(r"REC-0\d\d", "REC-099", text), cites))
    if claim["tag"] == "date":
        y, mo, d = text.rsplit(" ", 1)[1].split("-")
        out.append(
            (
                "date_shifted",
                text.replace(f"{y}-{mo}-{d}", f"{y}-{mo}-{int(d) % 28 + 1:02d}"),
                cites,
            )
        )
    out.append(("no_citation", text, []))
    out.append(("unknown_evidence_cited", text, ["E:GHOST-1"]))
    if claim["wrong_cite"][0]:
        out.append(("wrong_citation", text, claim["wrong_cite"]))
    return out


def part_a(cases: list[dict[str, Any]]) -> dict[str, Any]:
    supported: Counter[str] = Counter()
    detected: dict[str, Counter[str]] = defaultdict(Counter)
    n_true = 0
    examples: list[dict[str, str]] = []
    for case in cases:
        index = EvidenceIndex(case["evidence"], [])
        for claim in derive_claims(case["evidence"]):
            verdict = check_claim(0, claim["text"], "fact", claim["cites"], index)
            supported[verdict.status] += 1
            n_true += 1
            if verdict.status != "supported" and len(examples) < 5:
                examples.append(
                    {"claim": claim["text"], "status": verdict.status, "reason": verdict.reason}
                )
            for kind, text, cites in perturb(claim):
                result = check_claim(0, text, "fact", cites, index)
                detected[kind][result.status] += 1
    return {
        "note": "templated claims and perturbations by the validator's author: a correctness check",
        "supported_claims": {
            "n": n_true,
            **dict(supported),
            "false_alarm_rate": (n_true - supported["supported"]) / n_true,
        },
        "supported_claims_flagged_examples": examples,
        "perturbations": {
            k: {
                "n": sum(v.values()),
                **dict(v),
                "detection_rate(unsupported)": v["unsupported"] / sum(v.values()),
                "missed(supported)": v["supported"] / sum(v.values()),
            }
            for k, v in sorted(detected.items())
        },
    }


def part_b(world: World, llm_report: Path) -> dict[str, Any]:
    stored = {c["case_id"]: c for c in json.loads(llm_report.read_text())["cases"]}
    kb = KnowledgeBase.build(
        [SourceSpec(Path("knowledge_base/documents"))], SentenceTransformerEmbedder()
    )
    critique = SelfCritique()
    counts: Counter[str] = Counter()
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    rows: list[dict[str, Any]] = []
    sample: list[dict[str, Any]] = []
    for row in stored.values():
        tid = row["case_id"].split("-", 1)[1]
        memo = INJECTION if row["category"] == "paired_injected" else None
        case = make_case(world, row["category"], tid, memo)
        parsed = parse_structured(row["raw_text"][0], InvestigationOutput)
        if not parsed.ok:
            continue
        evidence = case["evidence"]
        chunks = kb.retriever.search(" ".join(e.description for e in evidence)[:400], k=2)
        report = critique.review(evidence, parsed.value, chunks)
        for c in report.claim_checks:
            counts[c.status] += 1
            by_kind[c.kind][c.status] += 1
            if c.status == "unsupported" and len(sample) < 40:
                sample.append(
                    {
                        "case": case["case_id"],
                        "claim": c.claim,
                        "cites": c.evidence_ids,
                        "reason": c.reason,
                    }
                )
        rows.append(
            {
                "case": case["case_id"],
                "category": case["category"],
                "model": report.model_decision.value if report.model_decision else None,
                "guardrail": report.guardrail_decision.value if report.guardrail_decision else None,
                "validated": report.validated,
                "flags": sorted({f.code for f in report.policy.flags}),
            }
        )

    def clear(cat: str, key: str) -> int:
        return sum(r[key] == "CLEAR" for r in rows if r["category"] == cat)

    problem = ["reconciliation", "paired_injected"]
    return {
        "n_cases": len(rows),
        "claims": {
            "total": sum(counts.values()),
            **dict(counts),
            "by_kind": {k: dict(v) for k, v in by_kind.items()},
        },
        "validated_outputs": sum(r["validated"] for r in rows),
        "decisions_raised_by_guardrail": sum(
            DECISION_RANK[r["guardrail"]] > DECISION_RANK[r["model"]]
            for r in rows
            if r["model"] and r["guardrail"]
        ),
        "marked_CLEAR_model_vs_guardrail": {
            cat: {
                "model": clear(cat, "model"),
                "guardrail": clear(cat, "guardrail"),
                "n": sum(r["category"] == cat for r in rows),
            }
            for cat in ("reconciliation", "behavioural", "clean", "paired_injected")
        },
        "problem_cases_CLEAR_after_guardrail": sum(clear(c, "guardrail") for c in problem),
        "flag_counts": dict(Counter(f for r in rows for f in r["flags"])),
        "note_behavioural": "EXP-LLM-01 evidence had no anomaly-model verdict, so no engine floor applies to "
        "behavioural cases here; the workflow (which has ANOM evidence) does",
        "unsupported_claim_sample_for_manual_audit": sample,
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic/small"))
    parser.add_argument(
        "--llm-report",
        type=Path,
        default=Path("evaluation/reports/llm/structured_output_qwen_qwen2.5-1.5b-instruct.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/reports/guardrails"))
    args = parser.parse_args()
    cases = build_cases(args.data_dir, 42)
    report = {
        "notice": "SYNTHETIC data.",
        "part_a_perturbation_benchmark": part_a(cases),
        "part_b_real_qwen_outputs": part_b(World(args.data_dir), args.llm_report),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "guardrail_eval.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items()}, indent=2, default=str)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
