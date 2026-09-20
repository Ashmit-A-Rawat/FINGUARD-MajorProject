PY := .venv/bin/python

.PHONY: install test lint typecheck run data pipeline kyc-benchmark
install:
	$(PY) -m pip install -e ".[dev]"
test:
	$(PY) -m pytest
lint:
	$(PY) -m ruff check .
typecheck:
	$(PY) -m mypy backend data_pipeline scripts kyc evaluation experiments
pipeline:
	$(PY) scripts/run_pipeline.py --preset small
data:
	$(PY) scripts/generate_synthetic_data.py --preset small
run:
	$(PY) -m uvicorn backend.app.main:app --reload --port 8000
kyc-benchmark:
	$(PY) experiments/kyc/run_kyc_benchmark.py --n-customers 4000 --seed 42
