"""Build the fine-tuning train / validation / eval sets from real workflow runs.

    python scripts/build_finetune_dataset.py --data-dir data/synthetic/small --out data/finetune

Leakage controls (all deliberate):
* TIME: train cases come from the 60-80% period, eval cases from the last 20%. The anomaly
  model behind the ANOM evidence was trained on the first 60%: no optimistic scores in either pool.
* CUSTOMERS: eval customers never appear in train/validation.
* INJECTION WORDINGS: train and eval use disjoint memo wordings.
* LABELS: ground-truth labels are stored for EVALUATION ONLY; the training target is derived
  from the evidence (see llm/fine_tuning/teacher.py).
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from agents.context import build_context
from agents.coordinator.workflow import CaseWorkflow
from backend.app.schemas.domain import Transaction
from knowledge_base.embeddings.embedder import SentenceTransformerEmbedder
from llm.fine_tuning.dataset import INJECTIONS_EVAL, INJECTIONS_TRAIN, CaseRecord, write_jsonl
from llm.inference.mock import MockLLMProvider

logger = logging.getLogger(__name__)


def truth_tables(data_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    ledger = pd.read_csv(data_dir / "ledger_labels.csv", keep_default_na=False)
    tx = pd.read_csv(data_dir / "transaction_labels.csv", keep_default_na=False)
    recon = dict(zip(ledger["transaction_id"], ledger["discrepancy_type"], strict=True))
    anomalies = tx[tx["is_anomaly"].astype(str).eq("True")]
    return recon, dict(zip(anomalies["transaction_id"], anomalies["anomaly_type"], strict=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic/small"))
    parser.add_argument("--out", type=Path, default=Path("data/finetune"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--train-per-class",
        type=int,
        nargs=3,
        default=[70, 70, 100],
        metavar=("RECON", "BEHAV", "CLEAN"),
    )
    parser.add_argument("--eval-per-class", type=int, default=8)
    parser.add_argument("--injection-train", type=int, default=60)
    args = parser.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(args.seed)

    ctx = build_context(args.data_dir, SentenceTransformerEmbedder(), MockLLMProvider())
    workflow = CaseWorkflow(ctx)
    txs = [t for v in ctx.store.transactions_by_customer.values() for t in v]
    stamps = np.array([t.timestamp for t in txs], dtype="datetime64[ns]")
    b1 = np.datetime64(ctx.anomaly.trained_until.to_datetime64())
    b2 = np.quantile(stamps.astype("int64"), 0.8)
    recon_truth, anomaly_truth = truth_tables(args.data_dir)
    problem_ids = set(recon_truth) | set(anomaly_truth)

    def pool(kind: str, start: np.datetime64 | None, stop: int | None) -> list[Transaction]:
        chosen: list[Transaction] = []
        for t in txs:
            ts = np.datetime64(t.timestamp, "ns")
            if start is not None and ts < start:
                continue
            if stop is not None and ts.astype("int64") >= stop:
                continue
            tid = t.transaction_id
            if (
                (kind == "reconciliation" and tid in recon_truth)
                or (kind == "behavioural" and tid in anomaly_truth and tid not in recon_truth)
                or (
                    kind == "clean"
                    and tid not in problem_ids
                    and len(ctx.store.transactions_by_customer[t.customer_id]) > 10
                )
            ):
                chosen.append(t)
        return chosen

    def pick(candidates: list[Transaction], n: int, banned: set[str]) -> list[Transaction]:
        order = rng.permutation(len(candidates))
        out: list[Transaction] = []
        used = set(banned)
        for i in order:
            t = candidates[int(i)]
            if t.customer_id not in used:  # one case per customer keeps cases independent
                out.append(t)
                used.add(t.customer_id)
            if len(out) == n:
                break
        return out

    def make(category: str, tx: Transaction, memo: str | None = None) -> CaseRecord:
        state = workflow.run(workflow.create_case(tx.customer_id, [tx.transaction_id]))
        evidence = [e.model_copy(deep=True) for e in state.evidence]
        if memo is not None:
            for e in evidence:
                if e.evidence_id == tx.transaction_id:
                    e.payload = {**e.payload, "payment_memo": memo}
        tid = tx.transaction_id
        truth_type = recon_truth.get(tid) or anomaly_truth.get(tid) or "none"
        return CaseRecord(
            case_id=state.case_id + ("" if memo is None else "+memo"),
            category=category,
            truth_problem=tid in problem_ids,
            truth_type=truth_type,
            evidence=[e.model_dump(mode="json") for e in evidence],
            chunks=[c.model_dump(mode="json") for c in state.knowledge_chunks],
            injected=memo,
            meta={"customer_id": tx.customer_id, "transaction_id": tid},
        )

    # -- eval first, so its customers can be excluded from training --
    eval_records: list[CaseRecord] = []
    eval_customers: set[str] = set()
    for category in ("reconciliation", "behavioural", "clean"):
        chosen = pick(
            pool(category, np.datetime64(int(b2), "ns"), None), args.eval_per_class, eval_customers
        )
        eval_customers |= {t.customer_id for t in chosen}
        eval_records += [make(category, t) for t in chosen]
    train_records: list[CaseRecord] = []
    for category, n in zip(
        ("reconciliation", "behavioural", "clean"), args.train_per_class, strict=True
    ):
        chosen = pick(pool(category, b1, int(b2)), n, eval_customers)
        train_records += [make(category, t) for t in chosen]
    # injection-augmented training cases: existing train transactions plus a TRAIN-wording memo
    base = [r for r in train_records if r.injected is None]
    for i in rng.permutation(len(base))[: args.injection_train]:
        r = base[int(i)]
        tx = next(
            t
            for t in ctx.store.transactions_by_customer[r.meta["customer_id"]]
            if t.transaction_id == r.meta["transaction_id"]
        )
        train_records.append(
            make(r.category, tx, INJECTIONS_TRAIN[int(rng.integers(0, len(INJECTIONS_TRAIN)))])
        )
    # injection eval: the reconciliation eval cases again, with EVAL wordings
    injected_eval = []
    for i, r in enumerate([r for r in eval_records if r.category == "reconciliation"]):
        tx = next(
            t
            for t in ctx.store.transactions_by_customer[r.meta["customer_id"]]
            if t.transaction_id == r.meta["transaction_id"]
        )
        injected_eval.append(make(r.category, tx, INJECTIONS_EVAL[i % len(INJECTIONS_EVAL)]))

    order = rng.permutation(len(train_records))
    shuffled = [train_records[int(i)] for i in order]
    n_val = max(20, len(shuffled) // 12)
    val, train = shuffled[:n_val], shuffled[n_val:]
    write_jsonl(train, args.out / "train.jsonl")
    write_jsonl(val, args.out / "val.jsonl")
    write_jsonl(eval_records, args.out / "eval.jsonl")
    write_jsonl(injected_eval, args.out / "eval_injected.jsonl")
    card = {
        "note": "SYNTHETIC. Targets come from evidence only; labels are for evaluation.",
        "seed": args.seed,
        "boundaries": {
            "anomaly_model_trained_until": str(b1),
            "eval_starts": str(np.datetime64(int(b2), "ns")),
        },
        "counts": {
            "train": len(train),
            "val": len(val),
            "eval": len(eval_records),
            "eval_injected": len(injected_eval),
        },
        "train_by_category": dict(Counter(r.category for r in train)),
        "train_injected": sum(r.injected is not None for r in train),
        "train_truth_problem_share": float(np.mean([r.truth_problem for r in train])),
        "eval_by_category": dict(Counter(r.category for r in eval_records)),
        "customers_overlap_train_eval": len(
            {r.meta["customer_id"] for r in train + val} & eval_customers
        ),
    }
    (args.out / "dataset_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
