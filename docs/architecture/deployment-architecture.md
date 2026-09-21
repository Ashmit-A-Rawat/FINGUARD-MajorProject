# Deployment Architecture

Status: Phase 14. `docker compose` runs PostgreSQL + API + web UI. The inference layer is hardware-agnostic and replaceable; no vendor-specific infrastructure code exists.


## Phase 11: running the API and UI

Environment (see `.env.example`): `DATABASE_URL` (default SQLite file; use PostgreSQL in production), `API_SECRET_KEY` (at least 32 random characters; required outside development),
`TOKEN_TTL_MINUTES`, `ENGINE_ENABLED`, `DATASET_DIR`, `LLM_PROVIDER` / `LLM_MODEL`, `CORS_ORIGINS`.

**Database roles (production).** Create the schema with an owner role, then run the API as a role that has `SELECT, INSERT` on `audit_events` and *no* `UPDATE` or `DELETE`. Together with the hash chain this makes tampering
detectable and, for the API's own credentials, impossible. Example: `GRANT SELECT, INSERT ON audit_events TO finguard_api; REVOKE UPDATE, DELETE ON audit_events FROM finguard_api;`
Schema is created with `create_all` on start; migrations (Alembic) are not set up yet.

**Reverse proxy.** Terminate TLS, serve `frontend/dist`, proxy `/api` to the API, and add a strict Content-Security-Policy. Keep the API on an internal network. The API process must run a **single instance** with the current in-process
job queue and login throttle.


## Phase 14: containers

```bash
export API_SECRET_KEY=$(openssl rand -base64 48)      # required; the API refuses to start in production without it
docker compose up --build                            # UI at http://localhost:8080
docker compose exec -e FINGUARD_PASSWORD='choose-one' api python scripts/create_user.py alice analyst
```

| Service | Image | Notes |
|---|---|---|
| `postgres` | postgres:16 | not published to the host; data in the `pgdata` volume |
| `api` | `Dockerfile` (python:3.11-slim, CPU torch, non-root, healthcheck) | single process on purpose (in-process job queue and login throttle); runs offline (`HF_HUB_OFFLINE=1`) |
| `web` | `frontend/Dockerfile` (node build, nginx) | serves the UI, proxies `/api` and `/health` to the API, strict CSP, no CORS needed |

- **Model weights are not in the image.** Put them under `./llm/models/<name>/` (mounted read-only). Set `LLM_PROVIDER=qwen`, `LLM_MODEL=llm/models/qwen2.5-0.5b-instruct`
  and, for the fine-tuned model, `LLM_ADAPTER=llm/fine_tuning/adapters/qwen0.5b-lora-v1`. The default `LLM_PROVIDER=mock` runs the whole system with clearly labelled mock outputs. The container is CPU-only (`LLM_DEVICE=cpu`); a GPU host needs the NVIDIA container runtime and a CUDA torch build.
- **Data.** The image generates the small SYNTHETIC dataset at build time (seeded). Replace it by mounting a dataset directory and setting `DATASET_DIR`.
- `SEMANTIC_TRIPWIRE=true` (default) enables the second, embedding-based injection detector in the API; it needs the MiniLM model already present in the image.
- **One build-time download**: the MiniLM embedding model. After the build the container needs no network.
- **TLS** is not handled here: terminate it in front of `web`. Do not publish the API or database ports.
- **Audit table permissions**: see the database-roles paragraph above; the compose file uses one owner role for simplicity, which is a development convenience, not the production setup.
- CI (`.github/workflows/ci.yml`) runs lint, types, tests and the frontend build on every push.
