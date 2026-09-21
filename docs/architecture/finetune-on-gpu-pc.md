# Fine-tuning on a separate machine (step by step)

**Easiest option: a free cloud notebook.** Open [notebooks/train_lora_colab.ipynb](../../notebooks/train_lora_colab.ipynb) in Google Colab (File > Upload notebook, or open it from GitHub), choose Runtime > Change runtime type > GPU (a free T4 works, roughly 30-60 minutes), and run the cells top to bottom. It clones the repo, downloads the base model, trains, evaluates and gives you one zip to download. Kaggle notebooks work the same way (turn on GPU and Internet). The free session is deleted when it ends, so download the zip before closing it. The trainer automatically uses float32 on GPUs without bfloat16 (a T4). The manual steps for your own PC follow.

The laptop (8 GB, Apple Silicon) cannot train: a single training sequence is ~2,000 tokens and MPS
runs out of memory. Everything else is built and wired; only training needs a bigger machine.
Needs an NVIDIA GPU with **>= 8 GB VRAM** (12 GB+ is comfortable). CPU-only works but takes many hours.

## 1. Get the code and the data
```bash
git clone https://github.com/Ashmit-A-Rawat/FINGUARD-MajorProject.git FINGUARD && cd FINGUARD
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA version
pip install -e ".[dev,data,api,ml,kyc,rag,llm]"
```
The training data (`data/finetune/*.jsonl`, synthetic, 1.2 MB) is already in the repo, so identical on every machine.

## 2. Put the SLM here
The model must be a folder containing `model.safetensors`, `config.json`, `tokenizer.json`, etc:

| What | Place it in |
|------|-------------|
| Qwen2.5-0.5B-Instruct (trained + evaluated) | `llm/models/qwen2.5-0.5b-instruct/` |
| Qwen2.5-1.5B-Instruct (optional reference arm) | `llm/models/qwen2.5-1.5b-instruct/` |

Download from Hugging Face (`Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`), e.g.
`huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir llm/models/qwen2.5-0.5b-instruct`
(or copy the folder from the laptop by USB). These folders are git-ignored.

## 3. Train (about 240 examples x 2 epochs)
```bash
export OMP_NUM_THREADS=1
python scripts/train_lora.py            # writes llm/fine_tuning/adapters/qwen0.5b-lora-v1/ + train_log.json
```
Quick check first: `python scripts/train_lora.py --max-steps 2 --out /tmp/probe`.
If it runs out of memory: `--grad-accum 4` does not help memory; lower `--max-len` cannot go below ~2,400
(the prompts are that long). Use a bigger GPU or `--rank 8`.

## 4. Evaluate (four arms, bootstrap CIs)
```bash
python experiments/llm/run_finetune_eval.py --reference-model llm/models/qwen2.5-1.5b-instruct
```
Writes `evaluation/reports/llm/finetune_eval.json`.

## 5. Bring the results back to the laptop
Copy back **either** just the report JSON (`evaluation/reports/llm/finetune_eval.json`) and
`train_log.json`, **or** the whole adapter folder `llm/fine_tuning/adapters/qwen0.5b-lora-v1/` (~35 MB) into the same
path on the laptop. Then tell Claude "adapter is in place"; the write-up and Phase 12 commit follow.

## 6. Use the adapter in the app
Set in `.env`:
```
LLM_PROVIDER=qwen
LLM_MODEL=llm/models/qwen2.5-0.5b-instruct
LLM_ADAPTER=llm/fine_tuning/adapters/qwen0.5b-lora-v1
```
Leave `LLM_ADAPTER` empty to run the plain base model. With `LLM_PROVIDER=mock` nothing is loaded.
