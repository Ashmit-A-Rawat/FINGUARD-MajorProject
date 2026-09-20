# FIN-GUARD API image (CPU). Model weights are NOT baked in: mount them at /app/llm/models (read-only).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app OMP_NUM_THREADS=1 \
    HF_HOME=/opt/hf-cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU-only PyTorch first (a much smaller layer than the default CUDA wheels)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY backend backend
COPY data_pipeline data_pipeline
COPY kyc kyc
COPY anomaly_detection anomaly_detection
COPY reconciliation reconciliation
COPY knowledge_base knowledge_base
COPY llm llm
COPY agents agents
COPY guardrails guardrails
COPY evaluation evaluation
COPY scripts scripts
RUN pip install --no-cache-dir ".[api,db,data,ml,kyc,rag,llm]"

# One deliberate build-time download of the small embedding model, so the running container needs no network.
RUN HF_HUB_OFFLINE=0 python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"
# Synthetic dataset the engines analyse (deterministic, seeded). Replace by mounting real data in production.
RUN python scripts/generate_synthetic_data.py --preset small

RUN useradd --create-home --uid 10001 finguard && chown -R finguard /app /opt/hf-cache
USER finguard
ENV DATASET_DIR=/app/data/synthetic/small
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD curl -fsS http://localhost:8000/health || exit 1
# One process only: the job queue and login throttle are in-process.
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
