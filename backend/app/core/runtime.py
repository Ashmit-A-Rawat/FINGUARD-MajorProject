"""Process-wide native-library settings. Import BEFORE torch, xgboost or scikit-learn.

Root cause of three separate crashes on macOS (segfaults and silent 0%-CPU deadlocks): PyTorch,
XGBoost and friends each ship their own OpenMP runtime, and two runtimes running multi-threaded
in one process clash. Limiting OpenMP to one thread makes every import order and every
combination safe (verified: torch-first, xgboost-first, and Chroma + sentence-transformers +
XGBoost + scikit-learn + PyTorch training in one process). Cost: no intra-op parallelism in
those libraries; use process-level parallelism instead.
"""

import os


def configure_native_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
