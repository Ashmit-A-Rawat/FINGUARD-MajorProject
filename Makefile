PY := .venv/bin/python

.PHONY: install test lint typecheck run data
install:
	$(PY) -m pip install -e ".[dev]"
test:
	$(PY) -m pytest
lint:
	$(PY) -m ruff check .
typecheck:
	$(PY) -m mypy backend data_pipeline scripts
data:
	$(PY) scripts/generate_synthetic_data.py --preset small
run:
	$(PY) -m uvicorn backend.app.main:app --reload --port 8000
