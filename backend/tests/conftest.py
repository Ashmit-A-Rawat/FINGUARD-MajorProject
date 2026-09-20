# Native-library thread limits must be set before anything imports torch / xgboost / sklearn.
from backend.app.core.runtime import configure_native_threads

configure_native_threads()
