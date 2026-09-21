"""Score the injection detectors on memo wordings: regex tripwire, semantic tripwire, and either.

Sets (evaluation/adversarial_dataset/):
  v1  development set for the regex patterns; also a training set for the semantic model
  v2  written before the regex change but seen while designing it; training set for the semantic model
  v3  written after the regex patterns were frozen (regex: held out); training set for the semantic model
  v4  written after the semantic configuration was fixed and never used to train anything: the ONLY set
      that is a fair held-out estimate for the semantic and combined detectors (regex: also held out)
Also counts false flags of the semantic detector on the free text of real (un-injected) case evidence.

    python experiments/adversarial/run_tripwire_eval.py --label after
"""

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from guardrails.policy_checks import free_text
from knowledge_base.embeddings.embedder import SentenceTransformerEmbedder
from knowledge_base.ingestion.sanitize import injection_flags
from knowledge_base.ingestion.semantic_tripwire import SemanticTripwire
from llm.fine_tuning.dataset import read_jsonl

ADV = Path("evaluation/adversarial_dataset")
SETS = {
    "v1": ADV / "memo_injections.json",
    "v2": ADV / "memo_injections_heldout_v2.json",
    "v3": ADV / "memo_injections_heldout_v3.json",
    "v4": ADV / "memo_injections_heldout_v4.json",
}
SEMANTIC_TRAINED_ON = {"v1", "v2", "v3"}
BENIGN_PREFIXES = ("benign", "hard_benign")


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    p, z = k / n, 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="after")
    ap.add_argument("--data", type=Path, default=Path("data/finetune"))
    ap.add_argument("--out", type=Path, default=Path("evaluation/reports/adversarial"))
    args = ap.parse_args()

    embedder = SentenceTransformerEmbedder()
    semantic = SemanticTripwire.load(embedder)
    detectors: dict[str, Callable[[str], bool]] = {"regex": lambda t: bool(injection_flags(t))}
    if semantic is not None:
        detectors["semantic"] = semantic.is_injection
        detectors["either"] = lambda t: bool(injection_flags(t)) or semantic.is_injection(t)

    report: dict[str, Any] = {
        "label": args.label,
        "semantic_available": semantic is not None,
        "sets": {},
    }
    lines = [f"# Tripwire evaluation ({args.label})\n"]
    for set_name, path in SETS.items():
        classes: dict[str, list[str]] = json.loads(path.read_text())["classes"]
        attacks = [w for c, ws in classes.items() if not c.startswith(BENIGN_PREFIXES) for w in ws]
        benign = [w for c, ws in classes.items() if c.startswith(BENIGN_PREFIXES) for w in ws]
        entry: dict[str, Any] = {"attacks": len(attacks), "benign": len(benign), "detectors": {}}
        lines.append(f"## {set_name} ({len(attacks)} attacks, {len(benign)} benign)")
        for name, detect in detectors.items():
            caught = [w for w in attacks if detect(w)]
            fps = [w for w in benign if detect(w)]
            in_training = name != "regex" and set_name in SEMANTIC_TRAINED_ON
            entry["detectors"][name] = {
                "recall": len(caught) / len(attacks),
                "recall_ci95": wilson(len(caught), len(attacks)),
                "caught": len(caught),
                "false_positives": len(fps),
                "fpr_ci95": wilson(len(fps), len(benign)),
                "in_training_set": in_training,
                "missed": [w for w in attacks if w not in caught],
                "false_positive_wordings": fps,
            }
            r = entry["detectors"][name]
            lines.append(
                f"- {name}{' (TRAINED ON THIS SET: not a held-out estimate)' if in_training else ''}: "
                f"recall {len(caught)}/{len(attacks)} = {r['recall']:.2f} "
                f"[{r['recall_ci95'][0]:.2f}, {r['recall_ci95'][1]:.2f}]; false positives "
                f"{len(fps)}/{len(benign)} [{r['fpr_ci95'][0]:.2f}, {r['fpr_ci95'][1]:.2f}]"
            )
        report["sets"][set_name] = entry
        lines.append("")

    if semantic is not None:
        seen: set[str] = set()
        texts = []
        for split in ("eval", "val", "train"):
            for rec in read_jsonl(args.data / f"{split}.jsonl"):
                key = rec.meta.get("transaction_id", rec.case_id)
                if rec.injected is None and key not in seen:
                    seen.add(key)
                    for e in rec.evidence_objects():
                        texts += list(free_text(e.payload))
        flagged = [t for t in texts if semantic.is_injection(t)]
        report["real_evidence_free_text"] = {
            "cases": len(seen),
            "strings": len(texts),
            "flagged": len(flagged),
        }
        lines.append(
            f"## real un-injected case evidence: {len(texts)} free-text strings in {len(seen)} cases, "
            f"{len(flagged)} flagged by the semantic detector"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"tripwire_eval_{args.label}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    (args.out / f"tripwire_eval_{args.label}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
