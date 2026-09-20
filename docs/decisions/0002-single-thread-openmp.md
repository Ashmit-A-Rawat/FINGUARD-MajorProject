# ADR 0002: Limit OpenMP to one thread process-wide

## Context
PyTorch, XGBoost, scikit-learn, ChromaDB and sentence-transformers each bundle or load their own OpenMP runtime
on macOS. When two runtimes both run multi-threaded in one process we saw, across Phases 5 and 7:
- segfaults (exit 139): importing `torch` before `xgboost`, and sentence-transformer inference after `xgboost`;
- silent deadlocks at 0% CPU with no error: PyTorch training after XGBoost / scikit-learn had used their thread pools.

Import-order tricks and `torch.set_num_threads(1)` alone were each insufficient in some combination.
`KMP_DUPLICATE_LIB_OK=TRUE` did not help.

## Decision
`backend/app/core/runtime.py:configure_native_threads()` sets `OMP_NUM_THREADS=1`. The `kyc`, `anomaly_detection` and
`knowledge_base` packages call it in their `__init__` (before any heavy import), and so does the test `conftest.py`.
Verified safe: torch-first, xgboost-first, and Chroma + sentence-transformers + XGBoost + scikit-learn + PyTorch
training in a single process.

## Consequences
- No intra-op parallelism in these libraries (embedding and neural training are slower; results become reproducible).
- Any new entry point must import one of these packages (or call the function) before importing torch / xgboost.
- Scale-out should use processes or separate services (for example the LLM server as its own process in Phase 8).
- Unrelated macOS quirk documented in the README: hidden `.pth` files inside `.venv` disable the editable install.
