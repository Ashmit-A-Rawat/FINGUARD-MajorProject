# Deployment Architecture

Status: Phase 1 baseline. docker-compose currently provides PostgreSQL only. The inference layer is hardware-agnostic and replaceable; no vendor-specific infrastructure code exists.


## Phase 11: running the API and UI

Environment (see `.env.example`): `DATABASE_URL` (default SQLite file; use PostgreSQL in production), `API_SECRET_KEY` (at least 32 random characters; required outside development),
`TOKEN_TTL_MINUTES`, `ENGINE_ENABLED`, `DATASET_DIR`, `LLM_PROVIDER` / `LLM_MODEL`, `CORS_ORIGINS`.

**Database roles (production).** Create the schema with an owner role, then run the API as a role that has `SELECT, INSERT` on `audit_events` and *no* `UPDATE` or `DELETE`. Together with the hash chain this makes tampering
detectable and, for the API's own credentials, impossible. Example: `GRANT SELECT, INSERT ON audit_events TO finguard_api; REVOKE UPDATE, DELETE ON audit_events FROM finguard_api;`
Schema is created with `create_all` on start; migrations (Alembic) are not set up yet.

**Reverse proxy.** Terminate TLS, serve `frontend/dist`, proxy `/api` to the API, and add a strict Content-Security-Policy. Keep the API on an internal network. The API process must run a **single instance** with the current in-process
job queue and login throttle.
