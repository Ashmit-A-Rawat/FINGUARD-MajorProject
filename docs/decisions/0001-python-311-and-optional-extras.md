# ADR 0001: Python 3.11 and optional dependency extras

Context: dev machine is Apple M2, 8 GB RAM, no CUDA; default python is 3.10.
Decision: target Python >=3.11 (venv built with 3.11). Heavy dependencies (torch, transformers, chromadb, ...) are optional extras in pyproject.toml, installed only in the phase that needs them.
Consequence: Phase 1 install is small; each later phase declares its own extra.
