# API reference (Phase 11)

Base path `/api`. JSON in and out. Interactive schema at `/docs` (FastAPI OpenAPI) when the server runs.
All data is SYNTHETIC. The API is **advisory only**: there is no endpoint that freezes an account, blocks a card, reverses a
posting, contacts a customer or files a report (a test enforces that no such route name exists).

## Authentication
`POST /api/auth/login` `{username, password}` returns `{access_token, role, ...}`. Send `Authorization: Bearer <token>`.
Tokens are HS256 JWTs (8 h by default). The **database is the source of truth** for role and active status on every request, so
deactivating a user or changing a role takes effect immediately, and a forged role claim is ignored.
Failed logins are throttled (5 per 5 minutes per client and username, then 429) and never reveal whether the username exists.

## Roles
| Capability | analyst | auditor | admin |
|---|---|---|---|
| List / view cases, evidence, reports | yes | yes | yes |
| See personal details (name, DOB, address) | **yes** (each view is audited) | no | no |
| Open a case, list candidates | yes | no | no |
| Sign off, propose / confirm / reject escalation | **yes** | no | no |
| Read the audit trail and its integrity check | yes | yes | yes |
| Create users | no | no | yes |

Separation of duties: administrators and auditors cannot decide cases.

## Endpoints
| Method and path | Role | Notes |
|---|---|---|
| `GET /health` | none | `engine_ready` tells whether the analysis engine has finished loading |
| `POST /api/auth/login`, `GET /api/auth/me` | none / any | |
| `POST /api/admin/users` | admin | username `[a-z0-9._-]{3,64}`, password at least 12 characters |
| `GET /api/cases?status=&limit=&offset=` | any | inbox: status, advisory and model decisions, validated, warnings |
| `POST /api/cases` `{customer_id, transaction_ids?, context_days?}` | analyst | returns 202; the workflow runs in a background worker; 409 if the case exists; 404/422 for bad input; 503 while the engine loads |
| `GET /api/cases/{id}` | any | full detail: customer/KYC, timeline, anomaly, reconciliation, evidence, retrieved passages, investigation, guardrail verdicts, report, sign-off |
| `GET /api/candidates` | analyst | recent held-out transactions that could become cases |
| `POST /api/cases/{id}/sign-off` `{decision, reason, acknowledge_advisory_override}` | analyst | see below |
| `POST /api/cases/{id}/sign-off/confirm` `{reason}` | analyst (different from proposer) | closes an escalation |
| `POST /api/cases/{id}/sign-off/reject` `{reason}` | analyst (different from proposer) | the case stays open |
| `GET /api/cases/{id}/audit` | any | events with hashes plus the chain check |

## Sign-off rules (enforced on the server; the UI only mirrors them)
- **Identity comes from the token**, never from the request body (`extra="forbid"` rejects a spoofed `reviewer` field).
- A reason is mandatory. The case must be awaiting review.
- Choosing **CLEAR against a REVIEW/ESCALATE advisory** returns 409 unless `acknowledge_advisory_override=true`; the override is written to the audit trail.
- **ESCALATE is two-step (four-eyes).** The first analyst *proposes* (the case stays open, `pending_escalation` is set). A *different authenticated analyst*
  confirms (case closed with both names) or rejects (case stays open). The proposer confirming their own proposal returns 403.
- Every step is an audit event.

## Case lifecycle in the API
`queued -> running -> done | failed` is the *job* state; `status` follows the workflow states (`human_review` while awaiting a decision, `closed` after sign-off).
A failed job is visible (`job_state: failed`, `job_error`), never silent. The workflow itself already fails safe to human review.

## Validation and errors
Ids must match `[A-Za-z0-9_.:-]{1,100}`; unknown fields are rejected; sizes are bounded (for example at most 20 transaction ids, reasons up to 2,000 characters).
Errors are `{"detail": ...}` with 401 (no/invalid token), 403 (role), 404, 409 (state conflict), 422 (validation), 429 (throttled), 503 (engine loading).
