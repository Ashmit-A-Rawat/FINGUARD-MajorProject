"""Compare supervised, unsupervised and neural anomaly detectors on a synthetic dataset.

    python experiments/anomaly/run_anomaly_benchmark.py --preset medium --seed 42

Protocol: chronological split 60/20/20 (train/val/test) with straddling-episode purge;
features are causal. Models are fit on train; validation is used ONLY for early stopping and to
choose each model's decision threshold (F1-maximising); every reported number is on TEST.
Unsupervised models never see labels during fitting (their threshold does use validation labels).
"""

import argparse
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from anomaly_detection.base import AnomalyModel
from anomaly_detection.dataset import FeatureTable, load_feature_table
from anomaly_detection.registry import create_models
from anomaly_detection.temporal.splits import temporal_split
from evaluation.metrics.anomaly import (
    budget_metrics,
    cluster_bootstrap,
    f1_optimal_threshold,
    operating_point,
    pr_auc,
    recall_by_type,
    roc_auc,
)


def _latency(model: AnomalyModel, table: FeatureTable, idx: np.ndarray) -> dict[str, float]:
    """Median ms to score 1000 rows in one batch, and one row at a time."""
    batch = idx[:1000]
    batch_times = []
    for _ in range(5):
        t = time.perf_counter()
        model.score(table, batch)
        batch_times.append(time.perf_counter() - t)
    single_times = []
    for row in idx[:100]:
        t = time.perf_counter()
        model.score(table, np.array([row]))
        single_times.append(time.perf_counter() - t)
    return {
        "ms_per_1000_rows_batch": 1000 * float(np.median(batch_times)),
        "ms_single_row": 1000 * float(np.median(single_times)),
    }


def run(
    preset: str, seed: int, data_dir: Path, only: list[str] | None, out_dir: Path
) -> dict[str, Any]:
    t0 = time.perf_counter()
    table = load_feature_table(data_dir)
    split = temporal_split(table.timestamps, table.episode_id)
    split.check_no_future_leakage(table.timestamps)
    y_val, y_test = table.labels[split.val], table.labels[split.test]
    print(
        f"features {table.features.frame.shape}; split train/val/test = "
        f"{len(split.train)}/{len(split.val)}/{len(split.test)}; purged {split.purged_episodes} "
        f"straddling episodes ({split.purged_rows} rows); test anomalies {int(y_test.sum())}"
    )

    results: dict[str, Any] = {}
    selected = {n: m for n, m in create_models(seed).items() if not only or n in only}
    for position, (name, model) in enumerate(selected.items(), start=1):
        filled = round(30 * (position - 1) / len(selected))
        print(
            f"[{'#' * filled}{'-' * (30 - filled)}] {position - 1}/{len(selected)} done; "
            f"training {name} ...",
            flush=True,
        )
        t = time.perf_counter()
        model.fit(table, split)
        fit_seconds = time.perf_counter() - t
        val_score, test_score = model.score(table, split.val), model.score(table, split.test)
        threshold = f1_optimal_threshold(y_val, val_score)
        clusters = table.customer_codes[split.test]
        results[name] = {
            "family": model.family,
            "feature_set": model.feature_set,
            "uses_labels_for_fitting": model.uses_labels,
            "fit_seconds": fit_seconds,
            "epochs_run": len(getattr(model, "history", [])) or None,
            "threshold_from_val": threshold,
            "test": {
                "pr_auc": pr_auc(y_test, test_score),
                "roc_auc": roc_auc(y_test, test_score),
                **operating_point(y_test, test_score, threshold),
                "budget_1pct": budget_metrics(y_test, test_score, 0.01),
                "budget_2pct": budget_metrics(y_test, test_score, 0.02),
            },
            "test_ci95": cluster_bootstrap(y_test, test_score, clusters, threshold, seed=seed),
            "test_recall_by_anomaly_type": recall_by_type(
                table.anomaly_type[split.test], test_score, threshold
            ),
            "latency": _latency(model, table, split.test),
        }
        r = results[name]["test"]
        print(
            f"{name:24s} PR-AUC={r['pr_auc']:.3f} ROC-AUC={r['roc_auc']:.3f} "
            f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} fit={fit_seconds:.0f}s",
            flush=True,
        )

    report = {
        "notice": "SYNTHETIC data; results do not represent real-bank performance.",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit-learn": version("scikit-learn"),
            "torch": version("torch"),
            "xgboost": version("xgboost"),
            "device": "cpu",
            "preset": preset,
            "seed": seed,
        },
        "data": {
            "rows": table.n,
            "n_features_engineered": table.features.frame.shape[1],
            "n_features_basic": len(table.features.basic_columns),
            "train_val_test_rows": [len(split.train), len(split.val), len(split.test)],
            "anomalies_train_val_test": [
                int(table.labels[s].sum()) for s in (split.train, split.val, split.test)
            ],
            "test_base_rate": float(y_test.mean()),
            "split_boundaries": [str(b) for b in split.boundaries],
            "purged_straddling_episodes": split.purged_episodes,
            "purged_rows": split.purged_rows,
        },
        "protocol": "chronological 60/20/20; thresholds from val; metrics on test; cluster bootstrap by customer",
        "models": results,
        "total_seconds": time.perf_counter() - t0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"anomaly_benchmark_{preset}_seed{seed}"
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2))
    (out_dir / f"{stem}.md").write_text(render_markdown(report))
    return report


def render_markdown(report: dict[str, Any]) -> str:
    d, env = report["data"], report["environment"]
    lines = [
        "# Anomaly detection benchmark report",
        "",
        f"> {report['notice']} Generated by `experiments/anomaly/run_anomaly_benchmark.py`; do not edit by hand.",
        "",
        f"Preset `{env['preset']}`, seed {env['seed']}, CPU. Protocol: {report['protocol']}.",
        f"Rows: {d['rows']}; train/val/test = {d['train_val_test_rows']}; anomalies = {d['anomalies_train_val_test']}; "
        f"test base rate {d['test_base_rate']:.4f} (a random scorer has PR-AUC about equal to this).",
        f"Purged {d['purged_straddling_episodes']} anomaly episodes ({d['purged_rows']} rows) straddling split boundaries.",
        "",
        "## Test results",
        "",
        "| model | family | features | PR-AUC [95% CI] | ROC-AUC | precision | recall | F1 [95% CI] | FPR | FNR | P@1% | R@1% | P@2% | R@2% |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in report["models"].items():
        t, ci = m["test"], m["test_ci95"]
        lines.append(
            f"| {name} | {m['family']} | {m['feature_set']} | {t['pr_auc']:.3f} [{ci['pr_auc'][0]:.3f}, {ci['pr_auc'][1]:.3f}] | "
            f"{t['roc_auc']:.3f} | {t['precision']:.3f} | {t['recall']:.3f} | {t['f1']:.3f} [{ci['f1'][0]:.3f}, {ci['f1'][1]:.3f}] | "
            f"{t['false_positive_rate']:.4f} | {1 - t['recall']:.3f} | {t['budget_1pct']['precision']:.3f} | "
            f"{t['budget_1pct']['recall']:.3f} | {t['budget_2pct']['precision']:.3f} | {t['budget_2pct']['recall']:.3f} |"
        )
    types = sorted({k for m in report["models"].values() for k in m["test_recall_by_anomaly_type"]})
    lines += [
        "",
        "## Recall by anomaly type at each model's threshold (test)",
        "",
        "| type | n | " + " | ".join(report["models"]) + " |",
        "|---|---|" + "---|" * len(report["models"]),
    ]
    for kind in types:
        first = next(iter(report["models"].values()))["test_recall_by_anomaly_type"].get(kind, {})
        cells = [
            f"{m['test_recall_by_anomaly_type'].get(kind, {}).get('recall', float('nan')):.2f}"
            for m in report["models"].values()
        ]
        lines.append(f"| {kind} | {int(first.get('n', 0))} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Cost",
        "",
        "| model | fit s | epochs | ms / 1000 rows (batch) | ms / single row |",
        "|---|---|---|---|---|",
    ]
    for name, m in report["models"].items():
        lines.append(
            f"| {name} | {m['fit_seconds']:.1f} | {m['epochs_run'] or '-'} | "
            f"{m['latency']['ms_per_1000_rows_batch']:.1f} | {m['latency']['ms_single_row']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=None, help="subset of model names")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/reports/anomaly"))
    args = parser.parse_args()
    data_dir = args.data_dir or Path("data/synthetic") / args.preset
    run(args.preset, args.seed, data_dir, args.models, args.output_dir)
    print((args.output_dir / f"anomaly_benchmark_{args.preset}_seed{args.seed}.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
