"""Transaction anomaly detection."""

from backend.app.core.runtime import configure_native_threads

configure_native_threads()  # must run before torch / xgboost / sklearn are imported
