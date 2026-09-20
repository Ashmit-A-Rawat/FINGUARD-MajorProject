# Reproducing every reported result

All data is SYNTHETIC and seeded. Numbers vary slightly by library version and CPU; reports record versions.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api,data,ml,kyc,rag,llm]"        # torch: pick the wheel for your machine
export OMP_NUM_THREADS=1                                # required on macOS (two OpenMP runtimes)
make test lint typecheck                                # 388 backend tests, ruff, mypy
```

| Result | Command | Approx. time | Report |
|---|---|---|---|
| EXP-KYC-01 (RQ1) | `make kyc-benchmark` (also `--seed 43`) | 2-5 min | `evaluation/reports/kyc/` |
| EXP-ANOM-01 (RQ2) | `make anomaly-benchmark` (also `--seed 43`) | 3 min | `evaluation/reports/anomaly/` |
| EXP-REC-01 | `make reconciliation-eval` | 1 min | `evaluation/reports/reconciliation/` |
| EXP-RAG-01 | `make rag-benchmark` | 1 min | `evaluation/reports/rag/` |
| EXP-ABL-01 | `make ablation` | seconds | `evaluation/reports/ablation/` |
| EXP-ADV-01 | `make adversarial` | seconds | `evaluation/reports/adversarial/` |
| EXP-GUARD-01 | `make guardrail-eval` (needs the EXP-LLM-01 report) | seconds | `evaluation/reports/guardrails/` |
| Base-model LLM arms (RQ3/RQ4) | `python experiments/llm/run_finetune_eval.py --out evaluation/reports/llm/finetune_eval_base_laptop.json` | ~15 min on an M-series laptop | `evaluation/reports/llm/` |
| RQ5 orchestration | `python experiments/agents/run_orchestration_ablation.py` | ~15 min | `evaluation/reports/agents/` |
| Fine-tuning + fine-tuned arms (RQ6) | [finetune-on-gpu-pc.md](../architecture/finetune-on-gpu-pc.md) | GPU machine | `evaluation/reports/llm/finetune_eval.json` |
| Consolidated summary | `make results-report` | seconds | `docs/research/results-summary.md` |

Model weights are never downloaded implicitly: fetch them deliberately (`llm/models/`, see `docs/architecture/llm-architecture.md`).
The fine-tuning dataset is committed (`data/finetune/`), or rebuild it with `make finetune-data` (needs the small synthetic dataset and the MiniLM model).
