"""Transaction anomaly detection.

IMPORT ORDER MATTERS: on macOS, importing PyTorch before XGBoost makes the process segfault
(two OpenMP runtimes clash; exit code 139). XGBoost must be imported first, so it is imported
here, before any submodule can pull in torch.
"""

import xgboost  # noqa: F401
