"""Run the KYC ablation benchmark and write a report generated from the actual run.

    python experiments/kyc/run_kyc_benchmark.py --n-customers 4000 --seed 42

Protocol: reranker fit on TRAIN queries; each system's decision threshold chosen on DEV
(maximising F1); all reported numbers are on TEST. Splits are disjoint by entity.
"""

import argparse
import json
import platform
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from evaluation.benchmarks.kyc_benchmark import BenchmarkQuery, build_kyc_benchmark
from evaluation.metrics.matching import (
    FlatOutcomes,
    RankedOutcome,
    best_threshold,
    bootstrap_ci,
    ranking_metrics,
    summarize,
)
from kyc.models import MatcherConfig, QueryIdentity
from kyc.reranking.reranker import LearnedReranker
from kyc.semantic.embedder import DEFAULT_MODEL, SentenceTransformerEmbedder
from kyc.signals import SignalStore
from kyc.systems import SYSTEMS, CandidateScorer

CHUNK = 256


def _chunks(items: Sequence[BenchmarkQuery]) -> Iterator[Sequence[BenchmarkQuery]]:
    for i in range(0, len(items), CHUNK):
        yield items[i : i + CHUNK]


def _slices(q: BenchmarkQuery) -> tuple[str, ...]:
    tags = []
    if q.has_duplicate_entity:
        tags.append("duplicate_entity")
    if q.has_namesake_negative:
        tags.append("namesake_negative")
    return tuple(tags)


def run(n_customers: int, seed: int, model_name: str, out_dir: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    bench = build_kyc_benchmark(n_customers, seed)
    customers = bench.customers
    ids = [c.customer.customer_id for c in customers]
    embedder = SentenceTransformerEmbedder(model_name)
    store = SignalStore(customers, embedder)
    scorer = CandidateScorer(customers, MatcherConfig())
    print(f"benchmark + index built in {time.perf_counter() - t0:.1f}s: {bench.composition()}")

    # -- fit the reranker on TRAIN queries only --
    features: list[np.ndarray] = []
    labels: list[bool] = []
    for chunk in _chunks(bench.split("train")):
        for q, sig in zip(
            chunk, store.compute([q.document.name.canonical for q in chunk]), strict=True
        ):
            idx, matrix = scorer.training_candidates(QueryIdentity.from_document(q.document), sig)
            features.append(matrix)
            labels.extend(ids[i] in q.gold_customer_ids for i in idx)
    reranker = LearnedReranker(seed).fit(np.vstack(features), np.array(labels, dtype=int))
    scorer.reranker = reranker
    print(f"reranker fit on {len(labels)} candidate pairs ({int(sum(labels))} positive)")

    # -- score DEV and TEST queries with every system --
    eval_queries = bench.split("dev") + bench.split("test")
    outcomes: dict[str, dict[str, list[RankedOutcome]]] = {
        s: {"dev": [], "test": []} for s in SYSTEMS
    }
    signal_time, system_time = 0.0, dict.fromkeys(SYSTEMS, 0.0)
    for chunk in _chunks(eval_queries):
        t = time.perf_counter()
        signals = store.compute([q.document.name.canonical for q in chunk])
        signal_time += time.perf_counter() - t
        for q, sig in zip(chunk, signals, strict=True):
            identity = QueryIdentity.from_document(q.document)
            for system in SYSTEMS:
                t = time.perf_counter()
                scored = scorer.score(system, identity, sig)
                system_time[system] += time.perf_counter() - t
                outcomes[system][q.split].append(
                    RankedOutcome(
                        q.gold_customer_ids,
                        [(ids[c.customer_idx], c.final_score) for c in scored],
                        q.variation_type,
                        _slices(q),
                    )
                )
    n_eval = len(eval_queries)

    # -- thresholds from DEV, metrics on TEST --
    report_systems: dict[str, Any] = {}
    for system in SYSTEMS:
        threshold = best_threshold(FlatOutcomes(outcomes[system]["dev"]))
        test = outcomes[system]["test"]
        flat = FlatOutcomes(test)
        tp, fp, fn = flat.counts(threshold)
        groups: dict[str, list[RankedOutcome]] = {}
        for o in test:
            for key in (f"variation:{o.category}", *(f"slice:{s}" for s in o.slices)):
                groups.setdefault(key, []).append(o)
        by_group = {}
        for key, group in sorted(groups.items()):
            gtp, gfp, gfn = FlatOutcomes(group).counts(threshold)
            by_group[key] = {"n_queries": len(group), **summarize(gtp, gfp, gfn)}
        report_systems[system] = {
            "threshold_from_dev": threshold,
            "test": {**summarize(tp, fp, fn), **ranking_metrics(test)},
            "test_ci95": bootstrap_ci(tp, fp, fn, seed=seed),
            "test_by_group": by_group,
            "latency_ms_per_query": {
                "shared_signals": 1000 * signal_time / n_eval,
                "system_scoring": 1000 * system_time[system] / n_eval,
            },
        }

    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "embedding_model": model_name,
            "seed": seed,
        },
        "notice": "SYNTHETIC benchmark; results do not represent real-bank performance.",
        "benchmark": bench.composition(),
        "protocol": "reranker fit on train; threshold per system chosen on dev; metrics on test",
        "config": MatcherConfig().model_dump(),
        "systems": report_systems,
        "total_seconds": time.perf_counter() - t0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"kyc_benchmark_n{n_customers}_seed{seed}"
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2))
    (out_dir / f"{stem}.md").write_text(render_markdown(report))
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# KYC benchmark report",
        "",
        f"> {report['notice']} Generated by `experiments/kyc/run_kyc_benchmark.py`; do not edit by hand.",
        "",
        f"Embedding model: `{report['environment']['embedding_model']}`, seed {report['environment']['seed']}. "
        f"Protocol: {report['protocol']}.",
        f"Benchmark: {report['benchmark']['n_customers']} customers, "
        f"{report['benchmark']['queries_per_split']} queries per split.",
        "",
        "## Test-set results (threshold chosen on dev)",
        "",
        "| system | thr | precision | recall | F1 [95% CI] | false-match (queries) | missed-match | MRR | top-1 | ms/query |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in report["systems"].items():
        t, ci = s["test"], s["test_ci95"]["f1"]
        ms = (
            s["latency_ms_per_query"]["system_scoring"]
            + s["latency_ms_per_query"]["shared_signals"]
        )
        lines.append(
            f"| {name} | {s['threshold_from_dev']:.2f} | {t['precision']:.3f} | {t['recall']:.3f} | "
            f"{t['f1']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {t['false_match_rate']:.3f} | "
            f"{t['missed_match_rate']:.3f} | {t['mrr']:.3f} | {t['top1_accuracy']:.3f} | {ms:.1f} |"
        )
    groups = sorted({g for s in report["systems"].values() for g in s["test_by_group"]})
    lines += [
        "",
        "## Recall by query category (test)",
        "",
        "| category | n | " + " | ".join(report["systems"]) + " |",
        "|---|---|" + "---|" * len(report["systems"]),
    ]
    for g in groups:
        first = next(iter(report["systems"].values()))["test_by_group"].get(g, {})
        cells = [
            f"{s['test_by_group'].get(g, {}).get('recall', float('nan')):.2f}"
            for s in report["systems"].values()
        ]
        lines.append(f"| {g} | {first.get('n_queries', '')} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## False-match rate by query category (test)",
        "",
        "| category | " + " | ".join(report["systems"]) + " |",
        "|---|" + "---|" * len(report["systems"]),
    ]
    for g in groups:
        cells = [
            f"{s['test_by_group'].get(g, {}).get('false_match_rate', float('nan')):.2f}"
            for s in report["systems"].values()
        ]
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-customers", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/reports/kyc"))
    args = parser.parse_args()
    report = run(args.n_customers, args.seed, args.model, args.output_dir)
    print((args.output_dir / f"kyc_benchmark_n{args.n_customers}_seed{args.seed}.md").read_text())
    return 0 if report else 1


if __name__ == "__main__":
    sys.exit(main())
