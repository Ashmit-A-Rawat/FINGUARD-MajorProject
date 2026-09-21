"""Train the semantic injection tripwire (knowledge_base/ingestion/tripwire_model.json).

Positives: attack memo wordings of sets v1, v2 and v3. Negatives: their benign / hard-benign memos
plus the opening sentence of every knowledge-base section (legitimate policy text). Set v4 is NEVER
used here: it is the held-out set. Hyper-parameters are fixed (C=10, threshold 0.7) and were chosen
before v4 was written.

    python scripts/train_tripwire.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from knowledge_base.embeddings.embedder import DEFAULT_MODEL, SentenceTransformerEmbedder
from knowledge_base.ingestion.loader import load_documents
from knowledge_base.ingestion.semantic_tripwire import MODEL_PATH

ADV = Path("evaluation/adversarial_dataset")
TRAIN_SETS = [
    "memo_injections.json",
    "memo_injections_heldout_v2.json",
    "memo_injections_heldout_v3.json",
]
C, THRESHOLD = 10.0, 0.7


def main() -> None:
    attacks: list[str] = []
    benign: list[str] = []
    digest = hashlib.sha256()
    for name in TRAIN_SETS:
        raw = (ADV / name).read_bytes()
        digest.update(raw)
        for cls, words in json.loads(raw)["classes"].items():
            (benign if cls.startswith(("benign", "hard_benign")) else attacks).extend(words)
    kb = [
        s.text.split(". ")[0][:120]
        for d in load_documents(Path("knowledge_base/documents")).documents
        for s in d.sections
    ]
    embedder = SentenceTransformerEmbedder()
    negatives = benign + kb
    x = np.vstack([embedder.encode(attacks), embedder.encode(negatives)])
    y = [1] * len(attacks) + [0] * len(negatives)
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced", random_state=0).fit(x, y)
    spec = {
        "note": "Semantic injection tripwire: logistic regression on embeddings. SYNTHETIC.",
        "embedding_model": DEFAULT_MODEL,
        "weights": [round(float(v), 6) for v in clf.coef_[0]],
        "bias": round(float(clf.intercept_[0]), 6),
        "threshold": THRESHOLD,
        "C": C,
        "n_attack": len(attacks),
        "n_benign_memos": len(benign),
        "n_kb_sentences": len(kb),
        "training_sets_sha256": digest.hexdigest(),
    }
    MODEL_PATH.write_text(json.dumps(spec, indent=1))
    print(f"wrote {MODEL_PATH}: {len(attacks)} attacks, {len(negatives)} negatives")


if __name__ == "__main__":
    main()
