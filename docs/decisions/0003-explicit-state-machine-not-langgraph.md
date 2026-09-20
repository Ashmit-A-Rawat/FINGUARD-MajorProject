# ADR 0003: An explicit state machine for orchestration, not LangGraph

## Context
The brief allows LangGraph "or a clean state-machine architecture" and says not to add LangChain/LangGraph unnecessarily. The workflow is
linear and fixed (ten states); nothing in it needs an LLM to decide what happens next; and orchestration must be deterministic and auditable.

## Decision
`agents/coordinator/workflow.py` holds a tuple of `Step(name, actor, from_status, to_status, run)`. The coordinator executes them in order,
checks the case is in the expected state, times each step and appends an audit event. No planner, no dynamic routing, no extra dependency.

## Consequences
- The whole control flow is readable in one place and unit-testable (order, failure, sign-off).
- Failures are handled uniformly: any step exception routes the case to HUMAN_REVIEW with a best-effort report; nothing is retried silently.
- If branching or parallel steps are ever needed (for example running KYC and reconciliation concurrently), LangGraph can be adopted then;
  the `Step` contracts (typed inputs/outputs per agent) map directly onto graph nodes.
