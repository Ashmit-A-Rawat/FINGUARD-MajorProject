# FIN-GUARD

An on-premise multi-agent AI system for automated KYC verification and financial transaction anomaly detection and remediation.

> **Research prototype. All data is SYNTHETIC. No real PII, no real financial actions, no regulatory-compliance claims.**

## Status

Phases 1-3 complete: skeleton, schemas + synthetic data generator, data pipeline (ingest, clean, normalize, consolidate, canonical cases). No ML, RAG, LLM or agents yet.

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
| Run ML | *Phase 4-5* |
| Frontend | *Phase 11* |
| Local LLM | *Phase 8* |
| Evaluation | *Phase 13* |

## Research questions

See [docs/research/research-questions.md](docs/research/research-questions.md).
