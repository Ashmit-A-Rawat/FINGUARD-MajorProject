# Fine-tuning design (Phase 12)

Goal (RQ6): does parameter-efficient fine-tuning improve domain investigation quality over prompting and RAG? Not assumed: it is measured (EXP-FT-01).

## Data (`data/finetune/`, SYNTHETIC, committed, built by `scripts/build_finetune_dataset.py`)
| file | cases | purpose |
|---|---|---|
| train.jsonl | 242 | training (incl. 54 with an injected memo, TRAIN wordings) |
| val.jsonl | 22 | validation loss per epoch |
| eval.jsonl | 24 | held-out (8 reconciliation, 8 behavioural, 8 clean) |
| eval_injected.jsonl | 8 | the reconciliation eval cases again with EVAL injection wordings |

Each record holds the real evidence and retrieved chunks produced by the real workflow (mock LLM, real embedder, real engines), the ground-truth label (evaluation only), and metadata.
- **Time split**: the anomaly model is trained on the earliest 60%; training cases come from 60-80% of time; eval cases from the last 20%.
- **Customer-disjoint**: no customer appears in both train/val and eval (checked: overlap 0). One case per customer in eval.
- **Injection wordings disjoint** between train and eval. (The third eval wording was written to evade the original regex tripwire; the improved regex of EXP-ADV-02 now catches it. This does not change what the model sees, only the floor's reasons.)

## Targets (`llm/fine_tuning/teacher.py`)
The target for a case is a deterministic function of its EVIDENCE only, never of the ground-truth label: findings cite `E:<id>`, decision REVIEW if the engine floor applies else CLEAR, never ESCALATE (a human judgement), fixed placeholder confidence (0.6 / 0.7, uncalibrated). Across 60 real cases every target has 0 unsupported claims and a grounded summary (tested).
Consequence: the tuned model learns the *format and grounding discipline* and to follow the engines' verdicts. It cannot learn to beat the engines, because its targets are the engines. RQ6 is therefore about format, grounding and robustness, not "better detection".

## Training (`llm/fine_tuning/train.py`, `scripts/train_lora.py`)
LoRA r=16, alpha=32, dropout 0.05 on q,k,v,o,gate,up,down; loss on assistant tokens only; bf16 base; gradient checkpointing; grad-accum 8; lr 2e-4 cosine; 2 epochs; prompts are the exact deployed prompt (RAG on/off at random, random fence nonce). QLoRA (4-bit) is not used (no Apple-Silicon support); prompts are ~2,000 tokens, which does not fit an 8 GB laptop, so training runs on a GPU machine: [finetune-on-gpu-pc.md](finetune-on-gpu-pc.md). Adapters are git-ignored.

## Serving
`LocalHFProvider(..., adapter=path)` loads the adapter with PEFT; `provider.use_adapter=False` runs the plain base model with the adapter disabled, so all arms share one set of base weights. `LLM_ADAPTER` selects it in the app.

## Known limits
Small data (242 cases), one seed, one base model; targets are machine-generated; synthetic evidence is more regular than real evidence, so format gains may be easier than in practice.
