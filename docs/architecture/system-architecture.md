# System Architecture

Status: **Phase 1 skeleton.** Sections are filled in as each phase lands.

## Layers (build order = dependency order)

1. Data: synthetic banking data, schemas, ingestion, consolidation (Phases 2-3)
2. Engines: KYC entity resolution, anomaly detection, reconciliation (Phases 4-6)
3. Knowledge: banking knowledge base + RAG retrieval (Phase 7)
4. LLM: provider abstraction (mock / Qwen / Mistral), structured output (Phase 8)
5. Agents: deterministic state-machine orchestration (Phase 9)
6. Guardrails: evidence validation, self-critique (Phase 10)
7. Human review UI + audit trail (Phase 11)

## Principles

- Deterministic engines produce evidence; the LLM only explains and investigates it.
- Every AI claim must cite evidence IDs; unsupported claims route to human REVIEW.
- Retrieved documents are DATA, never instructions.
- No real financial actions (e.g. account freezing) are ever taken automatically.
- All data is synthetic and labelled as such.
