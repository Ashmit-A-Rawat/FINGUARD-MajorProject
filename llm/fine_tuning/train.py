"""LoRA fine-tuning on Apple Metal (or CPU/CUDA): plain PyTorch loop over PEFT, no extra trainer library.

Design notes
* Loss is computed on the ASSISTANT TOKENS ONLY (the JSON investigation); the long prompt is masked.
* Base weights stay frozen; only LoRA adapters train (attention and MLP projections). They are loaded in
  bfloat16, or in float32 on a CUDA GPU without bfloat16 support (for example a Colab T4), where 16-bit
  training of the adapters would be unstable.
* Gradient checkpointing keeps activation memory small enough for an 8 GB machine.
* QLoRA (4-bit) is not used: bitsandbytes has no Apple-Silicon support. Plain LoRA is the honest
  equivalent here.
"""

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from llm.fine_tuning.dataset import read_jsonl, training_examples
from llm.inference.base import Message
from llm.inference.local_hf import resolve_device

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainConfig:
    base: str
    train_path: Path
    val_path: Path
    out_dir: Path
    epochs: int = 2
    lr: float = 2e-4
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    grad_accum: int = 8
    max_len: int = 2048
    seed: int = 0
    warmup_frac: float = 0.05
    max_steps: int | None = None  # stop early (used for timing probes)
    device: str = "auto"


def encode(
    tokenizer: Any, messages: list[Message], target: str, max_len: int
) -> tuple[list[int], list[int]] | None:
    """(input_ids, labels) with the prompt masked to -100; None if the example is too long."""
    prompt = tokenizer.apply_chat_template(
        [m.model_dump() for m in messages], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(target + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) > max_len:
        return None
    return prompt_ids + answer_ids, [-100] * len(prompt_ids) + answer_ids


def pick_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        return torch.float32
    return torch.bfloat16


def _batch(pairs: list[tuple[list[int], list[int]]], device: str) -> dict[str, torch.Tensor]:
    ids, labels = pairs[0]  # micro-batch of one: no padding, no wasted compute
    return {
        "input_ids": torch.tensor([ids], device=device),
        "labels": torch.tensor([labels], device=device),
    }


def _mps_gb() -> float:
    return torch.mps.driver_allocated_memory() / 1e9 if torch.backends.mps.is_available() else 0.0


def train(cfg: TrainConfig) -> dict[str, Any]:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base)
    started = time.perf_counter()

    def prepare(path: Path, seed: int) -> tuple[list[tuple[list[int], list[int]]], int]:
        encoded = [
            encode(tokenizer, m, t, cfg.max_len)
            for m, t in training_examples(read_jsonl(path), seed)
        ]
        kept = [e for e in encoded if e is not None]
        return kept, len(encoded) - len(kept)

    train_set, dropped_train = prepare(cfg.train_path, cfg.seed)
    val_set, dropped_val = prepare(cfg.val_path, cfg.seed + 1)
    lengths = [len(i) for i, _ in train_set]
    answer_tokens = [sum(1 for x in lab if x != -100) for _, lab in train_set]

    dtype = pick_dtype(device)
    base: Any = AutoModelForCausalLM.from_pretrained(cfg.base, dtype=dtype)
    model: Any = base.to(device)
    model.config.use_cache = False
    lora = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=0.0)

    steps_per_epoch = math.ceil(len(train_set) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs if cfg.max_steps is None else cfg.max_steps
    warmup = max(1, int(cfg.warmup_frac * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return cfg.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    def val_loss() -> float:
        model.eval()
        with torch.no_grad():
            losses = [float(model(**_batch([ex], device)).loss) for ex in val_set]
        model.train()
        return sum(losses) / len(losses)

    log: dict[str, Any] = {
        "config": {k: str(v) for k, v in asdict(cfg).items()},
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "trainable_params": n_trainable,
        "train_examples": len(train_set),
        "val_examples": len(val_set),
        "dropped_too_long": {"train": dropped_train, "val": dropped_val},
        "tokens": {
            "mean_sequence": sum(lengths) / len(lengths),
            "max_sequence": max(lengths),
            "mean_answer": sum(answer_tokens) / len(answer_tokens),
        },
        "val_loss_before": val_loss(),
        "steps": [],
        "epochs": [],
    }
    print(
        f"train {len(train_set)} (dropped {dropped_train}), val {len(val_set)}; mean seq "
        f"{log['tokens']['mean_sequence']:.0f} tokens; trainable {n_trainable / 1e6:.1f}M; "
        f"val loss before {log['val_loss_before']:.3f}",
        flush=True,
    )

    model.train()
    step = 0
    peak = 0.0
    for epoch in range(cfg.epochs):
        order = list(range(len(train_set)))
        random.Random(cfg.seed + epoch).shuffle(order)
        for start in range(0, len(order), cfg.grad_accum):
            group = order[start : start + cfg.grad_accum]
            total = 0.0
            for idx in group:
                loss = model(**_batch([train_set[idx]], device)).loss / len(group)
                loss.backward()
                total += float(loss)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            peak = max(peak, _mps_gb())
            log["steps"].append(
                {"step": step, "epoch": epoch + 1, "loss": total, "lr": lr_at(step - 1)}
            )
            elapsed = time.perf_counter() - started
            print(
                f"[{'#' * round(30 * step / total_steps):-<30}] step {step}/{total_steps} epoch {epoch + 1} "
                f"loss {total:.3f} | {elapsed / 60:.1f} min | mps {peak:.1f} GB",
                flush=True,
            )
            if cfg.max_steps is not None and step >= cfg.max_steps:
                break
        log["epochs"].append({"epoch": epoch + 1, "val_loss": val_loss()})
        print(f"epoch {epoch + 1} val loss {log['epochs'][-1]['val_loss']:.3f}", flush=True)
        if cfg.max_steps is not None and step >= cfg.max_steps:
            break
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.out_dir))
    size_mb = sum(f.stat().st_size for f in cfg.out_dir.glob("*.safetensors")) / 1e6
    log.update(
        train_minutes=(time.perf_counter() - started) / 60,
        peak_mps_gb=peak,
        adapter_mb=size_mb,
        total_steps=step,
    )
    (cfg.out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    return log
