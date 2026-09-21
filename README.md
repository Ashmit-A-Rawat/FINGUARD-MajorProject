# FIN-GUARD

An on-premise multi-agent AI system for automated KYC verification and financial transaction anomaly detection and remediation.

> **Research prototype. All data is SYNTHETIC. No real PII, no real financial actions, no regulatory-compliance claims.**

## Status

All 14 phases are built. Short running summary: [docs/PROGRESS.md](docs/PROGRESS.md); every result with its threats to validity: [docs/research/experiments.md](docs/research/experiments.md); generated cross-experiment table: [docs/research/results-summary.md](docs/research/results-summary.md); how to reproduce: [docs/research/reproduce.md](docs/research/reproduce.md).

**One step is deliberately left to a GPU machine (a free Colab notebook works: [notebooks/train_lora_colab.ipynb](notebooks/train_lora_colab.ipynb)):** training the LoRA adapter (RQ6). Everything else, including the evaluation script for the fine-tuned arms, is in place: [docs/architecture/finetune-on-gpu-pc.md](docs/architecture/finetune-on-gpu-pc.md). Until then RQ6 is reported as PENDING, not estimated. Real local inference needs a deliberate model download (see docs/architecture/llm-architecture.md).

## Architecture

Data → consolidation → KYC engine + anomaly engine → reconciliation → knowledge base / RAG → local LLM → agents → self-critique / evidence validation → human sign-off → report + audit trail. Details: [docs/architecture/system-architecture.md](docs/architecture/system-architecture.md).

## Setup

Requires Python >= 3.11, Node >= 20 (frontend, later), Docker (optional).

```bash
python3.11 -m venv .venv
make install            # installs core + dev dependencies
cp .env.example .env
```

## Running

| Task | Command |
|---|---|
| Backend | `make run` (http://localhost:8000/health) |
| Tests | `make test` |
| Lint / types | `make lint`, `make typecheck` |
| Postgres | `docker compose up -d postgres` |
| Generate data | `make data` or `python scripts/generate_synthetic_data.py --preset small` (see [data architecture](docs/architecture/data-architecture.md)) |
| Data pipeline | `make pipeline` |
| KYC benchmark / ablation | `make kyc-benchmark` (report in `evaluation/reports/kyc/`) |
| Knowledge-base retrieval benchmark | `make rag-benchmark` |
| Reconciliation eval | `make reconciliation-eval` |
| Anomaly benchmark | `make anomaly-benchmark` (report in `evaluation/reports/anomaly/`); progress: `scripts/run_status.sh <log>` |
| Containers (API + web + PostgreSQL) | `API_SECRET_KEY=$(openssl rand -base64 48) docker compose up --build`, see [deployment](docs/architecture/deployment-architecture.md) |
| Ablations / adversarial tests | `make ablation`, `make adversarial`, `make results-report` |
| Backend API | `make api` (needs a user: `FINGUARD_PASSWORD='...' python scripts/create_user.py alice analyst`); docs in [docs/api/api.md](docs/api/api.md) |
| Frontend | `make frontend-dev` (http://localhost:5173), tests: `make frontend-test` |
| Guardrail evaluation | `make guardrail-eval` |
| Multi-agent demo | `make agents-demo` (needs the local model) |
| Local LLM | `make hardware`; fetch weights deliberately (`scripts/fetch_weights.sh`); set `LLM_PROVIDER=qwen`, `LLM_MODEL=<dir or id>`; check with `make llm-check` |
| Evaluation | *Phase 13* |

## Research questions

See [docs/research/research-questions.md](docs/research/research-questions.md).

## Troubleshooting (macOS)

- `ModuleNotFoundError: No module named 'backend'` when running a script: macOS marks files inside dot-directories
  (like `.venv`) as *hidden*, and Python silently ignores hidden `.pth` files, so the editable install stops working.
  Use `make` targets (they set `PYTHONPATH=.`), run `PYTHONPATH=. python scripts/...`, or run `chflags -R nohidden .venv`.
- Crashes (exit code 139) or silent hangs at 0% CPU when PyTorch, XGBoost, scikit-learn or ChromaDB share a process:
  caused by two OpenMP runtimes. `backend/app/core/runtime.py` limits OpenMP to one thread; it is applied automatically
  by the `kyc`, `anomaly_detection` and `knowledge_base` packages. Do not import `torch` before them in a new entry point.
