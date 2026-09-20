PY := PYTHONPATH=. .venv/bin/python

.PHONY: install test lint typecheck run data pipeline kyc-benchmark anomaly-benchmark reconciliation-eval rag-benchmark hardware llm-check agents-demo guardrail-eval
install:
	$(PY) -m pip install -e ".[dev]"
test:
	$(PY) -m pytest
lint:
	$(PY) -m ruff check .
typecheck:
	$(PY) -m mypy backend data_pipeline scripts kyc evaluation experiments anomaly_detection reconciliation knowledge_base llm guardrails agents
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
hardware:
	$(PY) scripts/assess_hardware.py
llm-check:
	$(PY) experiments/llm/run_structured_output_check.py --provider qwen --model llm/models/qwen2.5-1.5b-instruct
agents-demo:
	$(PY) experiments/agents/run_workflow_demo.py --provider qwen --model llm/models/qwen2.5-1.5b-instruct
guardrail-eval:
	$(PY) experiments/guardrails/run_guardrail_eval.py
