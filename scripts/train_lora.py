"""Fine-tune a local model with LoRA on the synthetic investigation dataset.

    python scripts/train_lora.py --base llm/models/qwen2.5-0.5b-instruct \
        --out llm/fine_tuning/adapters/qwen0.5b-lora-v1
"""

import argparse
import sys
from pathlib import Path

from llm.fine_tuning.train import TrainConfig, train


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="llm/models/qwen2.5-0.5b-instruct")
    p.add_argument("--data", type=Path, default=Path("data/finetune"))
    p.add_argument("--out", type=Path, default=Path("llm/fine_tuning/adapters/qwen0.5b-lora-v1"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None)
    a = p.parse_args()
    cfg = TrainConfig(
        base=a.base,
        train_path=a.data / "train.jsonl",
        val_path=a.data / "val.jsonl",
        out_dir=a.out,
        epochs=a.epochs,
        lr=a.lr,
        rank=a.rank,
        alpha=2 * a.rank,
        grad_accum=a.grad_accum,
        max_len=a.max_len,
        seed=a.seed,
        max_steps=a.max_steps,
    )
    log = train(cfg)
    print(
        f"done: {log['train_minutes']:.1f} min, peak {log['peak_mps_gb']:.1f} GB, "
        f"adapter {log['adapter_mb']:.0f} MB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
