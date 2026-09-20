# Import XGBoost before anything can import PyTorch (segfault on macOS otherwise; see
# anomaly_detection/__init__.py).
import xgboost  # noqa: F401
