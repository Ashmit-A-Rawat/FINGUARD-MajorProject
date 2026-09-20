"""Component ablation of the deterministic engine floor (no language model involved).

A case is FLAGGED when the engine floor is REVIEW. Each configuration keeps only some engines'
evidence and/or changes a policy switch, so the drop in recall (or rise in false flags) versus the
full system is that component's contribution on this data.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from backend.app.schemas.domain import Evidence, Severity
from evaluation.metrics.retrieval import bootstrap_mean_ci
from guardrails.policy_checks import PolicyConfig, engine_floor
from llm.fine_tuning.dataset import CaseRecord

ENGINE_PREFIX = {"recon": "REC-", "anomaly": "ANOM-", "kyc": "KYCM-"}
ALL_PREFIXES = tuple(ENGINE_PREFIX.values())


@dataclass
class Config:
    name: str
    engines: tuple[str, ...] = ("recon", "anomaly", "kyc")
    policy: PolicyConfig = field(default_factory=PolicyConfig)


def configs() -> list[Config]:
    p = PolicyConfig
    return [
        Config("full (all engines, default policy)"),
        Config("reconciliation only", ("recon",)),
        Config("anomaly model only", ("anomaly",)),
        Config("KYC matcher only", ("kyc",)),
        Config("reconciliation + anomaly", ("recon", "anomaly")),
        Config("reconciliation + KYC", ("recon", "kyc")),
        Config("anomaly + KYC", ("anomaly", "kyc")),
        Config("full, LOW-severity findings also count", policy=p(floor_min_severity=Severity.LOW)),
        Config(
            "full, only HIGH-severity findings count", policy=p(floor_min_severity=Severity.HIGH)
        ),
        Config(
            "full, settlement-status rules also count", policy=p(status_rules_force_review=True)
        ),
    ]


def _keep(evidence: Sequence[Evidence], engines: tuple[str, ...]) -> list[Evidence]:
    prefixes = tuple(ENGINE_PREFIX[e] for e in engines)
    return [
        e
        for e in evidence
        if e.evidence_id.startswith(prefixes) or not e.evidence_id.startswith(ALL_PREFIXES)
    ]


def flagged(record: CaseRecord, cfg: Config) -> bool:
    floor, _ = engine_floor(_keep(record.evidence_objects(), cfg.engines), cfg.policy)
    return floor is not None


def _ci(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    if not len(arr):
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    lo, hi = bootstrap_mean_ci(arr)
    return {"mean": float(arr.mean()), "lo": lo, "hi": hi, "n": len(arr)}


def evaluate(
    records: Sequence[CaseRecord], flag: Callable[[CaseRecord], bool]
) -> dict[str, dict[str, float]]:
    """Recall per problem category, false-flag rate on clean cases, overall precision."""
    flags = [(r, flag(r)) for r in records]
    out = {
        "recall_reconciliation": _ci(
            [float(f) for r, f in flags if r.category == "reconciliation"]
        ),
        "recall_behavioural": _ci([float(f) for r, f in flags if r.category == "behavioural"]),
        "false_flag_rate_clean": _ci([float(f) for r, f in flags if r.category == "clean"]),
    }
    tp = sum(f and r.truth_problem for r, f in flags)
    fp = sum(f and not r.truth_problem for r, f in flags)
    out["precision"] = {"mean": tp / (tp + fp) if tp + fp else float("nan"), "n": tp + fp}
    return out
