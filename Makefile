PY := .venv/bin/python

.PHONY: install test lint typecheck run health
install:
	$(PY) -m pip install -e ".[dev]"
test:
	$(PY) -m pytest
lint:
	$(PY) -m ruff check .
typecheck:
	$(PY) -m mypy backend
run:
	$(PY) -m uvicorn backend.app.main:app --reload --port 8000
