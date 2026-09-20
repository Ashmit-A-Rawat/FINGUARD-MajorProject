PY := PYTHONPATH=. .venv/bin/python

.PHONY: install test lint typecheck run data pipeline kyc-benchmark anomaly-benchmark reconciliation-eval rag-benchmark
install:
	$(PY) -m pip install -e ".[dev]"
test:
	$(PY) -m pytest
lint:
	$(PY) -m ruff check .
typecheck:
	$(PY) -m mypy backend data_pipeline scripts kyc evaluation experiments anomaly_detection reconciliation knowledge_base
pipeline:
	$(PY) scripts/run_pipeline.py --preset small
data:
	$(PY) scripts/generate_synthetic_data.py --preset small
run:
	$(PY) -m uvicorn backend.app.main:app --reload --port 8000
kyc-benchmark:
	$(PY) experiments/kyc/run_kyc_benchmark.py --n-customers 4000 --seed 42
anomaly-benchmark:
	$(PY) scripts/generate_synthetic_data.py --preset medium
	$(PY) experiments/anomaly/run_anomaly_benchmark.py --preset medium --seed 42
reconciliation-eval:
	$(PY) experiments/reconciliation/run_reconciliation_eval.py --preset medium
rag-benchmark:
	$(PY) experiments/rag/run_retrieval_benchmark.py
