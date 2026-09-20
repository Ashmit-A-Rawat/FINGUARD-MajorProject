"""Knowledge-base retrieval benchmark: retrieval modes x chunk sizes, abstention signal, injection
tripwire and a poisoning test. Generates evaluation/reports/rag/retrieval_benchmark.{json,md}.

    python experiments/rag/run_retrieval_benchmark.py

The questions and the documents were written by the same author, and there are only 43 answerable
questions, so intervals are wide and vocabulary overlap may flatter lexical retrieval.
"""

import argparse
import json
import sys
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.metrics.retrieval import (
    bootstrap_mean_ci,
    first_hit_rank,
    hit_at_k,
    ndcg_at_k,
    reciprocal_rank,
    separation_auroc,
)
from knowledge_base.chunking.chunker import ChunkingConfig
from knowledge_base.embeddings.embedder import DEFAULT_MODEL, SentenceTransformerEmbedder
from knowledge_base.service import KnowledgeBase, SourceSpec

DOCS = Path("knowledge_base/documents")
ADV_DOCS = Path("evaluation/adversarial_dataset/kb_injection")
GOLDEN = Path("evaluation/golden_dataset/kb_retrieval_questions.json")
ADV_Q = Path("evaluation/adversarial_dataset/kb_injection_questions.json")
CHUNK_CONFIGS = {
    "small_40w": ChunkingConfig(max_words=40, overlap_words=10, min_words=10),
    "default_120w": ChunkingConfig(max_words=120, overlap_words=20),
    "large_240w": ChunkingConfig(max_words=240, overlap_words=30),
}
MODES = ("bm25", "dense", "hybrid")


def _ranked(
    kb: KnowledgeBase, question: str, mode: str, k: int, include_untrusted: bool = False
) -> list[Any]:
    return kb.retriever.search(question, k=k, mode=mode, include_untrusted=include_untrusted)  # type: ignore[arg-type]


def _top1_scores(
    kb: KnowledgeBase, questions: list[dict[str, Any]], mode: str, key: str
) -> list[float]:
    return [_ranked(kb, q["question"], mode, 1)[0].component_scores[key] for q in questions]


def _ci_text(metrics: dict[str, Any], key: str) -> str:
    low, high = metrics[f"{key}_ci95"]
    return f"[{low:.2f}, {high:.2f}]"


def _evaluate_mode(
    kb: KnowledgeBase, questions: list[dict[str, Any]], mode: str, include_untrusted: bool = False
) -> dict[str, Any]:
    per_q: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for q in questions:
        gold = {(d, s) for d, s in q["gold"]}
        ranked = [
            (r.document_id, r.metadata["section_id"])
            for r in _ranked(kb, q["question"], mode, 10, include_untrusted)
        ]
        rank, doc_rank = first_hit_rank(ranked, gold), first_hit_rank(ranked, gold, doc_level=True)
        row = {
            "hit@1": hit_at_k(rank, 1),
            "hit@3": hit_at_k(rank, 3),
            "hit@5": hit_at_k(rank, 5),
            "doc_hit@5": hit_at_k(doc_rank, 5),
            "mrr": reciprocal_rank(rank),
            "ndcg@5": ndcg_at_k(ranked, gold, 5),
        }
        for key, value in row.items():
            per_q["all"][key].append(value)
            per_q[q["type"]][key].append(value)
    out: dict[str, Any] = {}
    for group, metrics in per_q.items():
        out[group] = {"n": len(metrics["hit@1"])}
        for key, values in metrics.items():
            arr = np.array(values)
            out[group][key] = float(arr.mean())
            if group == "all" and key in ("hit@1", "hit@5", "mrr"):
                out[group][f"{key}_ci95"] = bootstrap_mean_ci(arr)
    return out


def run(out_dir: Path) -> dict[str, Any]:
    embedder = SentenceTransformerEmbedder()
    golden = json.loads(GOLDEN.read_text())["questions"]
    answerable = [q for q in golden if q["gold"]]
    unanswerable = [q for q in golden if not q["gold"]]
    report: dict[str, Any] = {
        "notice": "SYNTHETIC knowledge base and hand-written questions (single author).",
        "embedding_model": DEFAULT_MODEL,
        "versions": {
            "chromadb": version("chromadb"),
            "sentence-transformers": version("sentence-transformers"),
        },
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
    }

    # -- 1. retrieval ablation: chunk size x mode --
    ablation: dict[str, Any] = {}
    kbs: dict[str, KnowledgeBase] = {}
    for name, cfg in CHUNK_CONFIGS.items():
        kb = KnowledgeBase.build([SourceSpec(DOCS)], embedder, cfg)
        kbs[name] = kb
        ablation[name] = {"n_chunks": len(kb.chunks)}
        for mode in MODES:
            ablation[name][mode] = _evaluate_mode(kb, answerable, mode)
            a = ablation[name][mode]["all"]
            print(
                f"{name:13s} {mode:7s} hit@1={a['hit@1']:.3f} hit@5={a['hit@5']:.3f} mrr={a['mrr']:.3f}",
                flush=True,
            )
    report["ablation"] = ablation

    # -- 2. abstention signal on the default index --
    kb = kbs["default_120w"]
    signals: dict[str, Any] = {}
    for label, mode, key in (
        ("dense_cosine", "dense", "dense"),
        ("bm25_normalised", "bm25", "bm25"),
    ):
        ans = _top1_scores(kb, answerable, mode, key)
        una = _top1_scores(kb, unanswerable, mode, key)
        signals[label] = {
            "mean_top1_answerable": float(np.mean(ans)),
            "mean_top1_unanswerable": float(np.mean(una)),
            "max_top1_unanswerable": float(np.max(una)),
            "min_top1_answerable": float(np.min(ans)),
            "auroc_answerable_vs_unanswerable": separation_auroc(ans, una),
        }
    report["abstention_signal_default_index"] = signals

    # -- 3. injection tripwire + cleaning report --
    adv = KnowledgeBase.build([SourceSpec(DOCS), SourceSpec(ADV_DOCS, "untrusted")], embedder)
    adv_spec = json.loads(ADV_Q.read_text())
    flagged_docs = sorted({c.document_id for c in adv.chunks if c.injection_flags})
    report["injection_tripwire"] = {
        "flagged_documents": flagged_docs,
        "expected_flagged": adv_spec["expected_injection_flag_docs"],
        "missed_expected": sorted(
            set(adv_spec["expected_injection_flag_docs"]) - set(flagged_docs)
        ),
        "misinformation_not_flagged_as_expected": sorted(
            set(adv_spec["expected_unflagged_misinformation_docs"]) - set(flagged_docs)
        ),
        "false_positive_chunks_in_trusted_docs": sum(
            1 for c in adv.chunks if c.trust == "trusted" and c.injection_flags
        ),
        "trusted_chunks": sum(c.trust == "trusted" for c in adv.chunks),
    }

    # -- 4. poisoning: does an untrusted document displace the right answer? --
    # Two caller policies: opt in to untrusted chunks (worst case) vs the retriever default (excluded).
    poison: dict[str, Any] = {}
    for policy, include in (("untrusted_included", True), ("default_trusted_only", False)):
        poison[policy] = {}
        for mode in MODES:
            rows = []
            for q in adv_spec["questions"]:
                gold = {(d, s) for d, s in q["gold"]}
                results = _ranked(adv, q["question"], mode, 5, include_untrusted=include)
                ranked = [(r.document_id, r.metadata["section_id"]) for r in results]
                gold_rank = first_hit_rank(ranked, gold)
                poison_ranks = [i for i, r in enumerate(results, 1) if r.document_id == q["poison"]]
                rows.append(
                    {
                        "id": q["id"],
                        "gold_rank": gold_rank,
                        "poison_rank": poison_ranks[0] if poison_ranks else None,
                        "poison_outranks_gold": bool(poison_ranks)
                        and (gold_rank is None or poison_ranks[0] < gold_rank),
                        "top1_trust": results[0].metadata["trust"] if results else None,
                    }
                )
            clean = _evaluate_mode(kbs["default_120w"], answerable, mode)["all"]
            poisoned = _evaluate_mode(adv, answerable, mode, include_untrusted=include)["all"]
            poison[policy][mode] = {
                "adversarial_questions": rows,
                "poison_in_top5": sum(r["poison_rank"] is not None for r in rows),
                "poison_outranks_gold": sum(r["poison_outranks_gold"] for r in rows),
                "top1_untrusted": sum(r["top1_trust"] == "untrusted" for r in rows),
                "adversarial_gold_hit@5": sum(r["gold_rank"] is not None for r in rows),
                "golden_hit@5_clean_index": clean["hit@5"],
                "golden_hit@5_with_poison_docs": poisoned["hit@5"],
            }
    report["poisoning"] = poison
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_benchmark.json").write_text(json.dumps(report, indent=2))
    (out_dir / "retrieval_benchmark.md").write_text(render(report))
    return report


def render(r: dict[str, Any]) -> str:
    L = [
        "# Knowledge-base retrieval benchmark",
        "",
        f"> {r['notice']} Generated by `experiments/rag/run_retrieval_benchmark.py`; do not edit by hand.",
        "",
        f"Embedding model `{r['embedding_model']}`; {r['n_answerable']} answerable + {r['n_unanswerable']} unanswerable questions.",
        "",
        "## Retrieval: chunk size x mode (answerable questions; section-level hits)",
        "",
        "| chunking | chunks | mode | hit@1 [95% CI] | hit@3 | hit@5 [95% CI] | doc hit@5 | MRR [95% CI] | nDCG@5 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, a in r["ablation"].items():
        for mode in MODES:
            m = a[mode]["all"]
            L.append(
                f"| {name} | {a['n_chunks']} | {mode} | {m['hit@1']:.3f} {_ci_text(m, 'hit@1')} | {m['hit@3']:.3f} | {m['hit@5']:.3f} {_ci_text(m, 'hit@5')} | {m['doc_hit@5']:.3f} | {m['mrr']:.3f} {_ci_text(m, 'mrr')} | {m['ndcg@5']:.3f} |"
            )
    types = sorted(k for k in r["ablation"]["default_120w"]["hybrid"] if k != "all")
    L += [
        "",
        "## hit@5 by question type (default chunking)",
        "",
        "| type | n | bm25 | dense | hybrid |",
        "|---|---|---|---|---|",
    ]
    for t in types:
        n = r["ablation"]["default_120w"]["hybrid"][t]["n"]
        L.append(
            f"| {t} | {n} | "
            + " | ".join(f"{r['ablation']['default_120w'][m][t]['hit@5']:.2f}" for m in MODES)
            + " |"
        )
    L += [
        "",
        "## Abstention signal (top-1 score, answerable vs unanswerable; default chunking)",
        "",
        "| score | mean answerable | mean unanswerable | max unanswerable | min answerable | AUROC |",
        "|---|---|---|---|---|---|",
    ]
    for k, s in r["abstention_signal_default_index"].items():
        L.append(
            f"| {k} | {s['mean_top1_answerable']:.3f} | {s['mean_top1_unanswerable']:.3f} | {s['max_top1_unanswerable']:.3f} | {s['min_top1_answerable']:.3f} | {s['auroc_answerable_vs_unanswerable']:.3f} |"
        )
    t = r["injection_tripwire"]
    L += [
        "",
        "## Injection tripwire (heuristic flags)",
        "",
        f"- Flagged documents: {t['flagged_documents']}; expected: {t['expected_flagged']}; missed: {t['missed_expected'] or 'none'}",
        f"- Misinformation documents (no imperative text) correctly NOT flagged, i.e. undetectable by this heuristic: {t['misinformation_not_flagged_as_expected']}",
        f"- False-positive chunks in the {t['trusted_chunks']} trusted chunks: {t['false_positive_chunks_in_trusted_docs']}",
        "",
        "## Poisoning: untrusted, topically matching documents added to the index",
        "",
        "`untrusted_included` = the caller opts in to untrusted chunks (worst case). "
        "`default_trusted_only` = retriever default.",
        "",
        "| policy | mode | poison in top-5 (of 6) | poison outranks gold | top-1 untrusted | gold in top-5 (of 6) | golden hit@5 clean | golden hit@5 with poison docs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for policy, by_mode in r["poisoning"].items():
        for mode, p in by_mode.items():
            L.append(
                f"| {policy} | {mode} | {p['poison_in_top5']} | {p['poison_outranks_gold']} | {p['top1_untrusted']} | "
                f"{p['adversarial_gold_hit@5']} | {p['golden_hit@5_clean_index']:.3f} | {p['golden_hit@5_with_poison_docs']:.3f} |"
            )
    return "\n".join(L) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/reports/rag"))
    args = parser.parse_args()
    run(args.output_dir)
    print((args.output_dir / "retrieval_benchmark.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
