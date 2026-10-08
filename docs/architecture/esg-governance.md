# ESG governance-risk indicator

Added in response to review feedback asking us to explore incorporating an ESG (Environmental,
Social, Governance) score as an additional evidence parameter. Code: `esg/governance_scoring.py`.
Tests: `backend/tests/esg/` (14 tests). Demo: `experiments/esg/run_governance_demo.py`.

## Scope, stated honestly

This implements the **Governance pillar only**, as two standard compliance heuristics computed
from data the system already has:

1. **New cross-border counterparty** — the transaction moves money to or from a country this
   specific customer has never transacted with before. An unexpected new cross-border counterparty
   is a standard enhanced-due-diligence trigger in real compliance practice.
2. **Possible structuring** — several of the customer's transactions, within a trailing window,
   sit just under a round reporting-style threshold: the classic "structuring"/"smurfing" pattern
   used to avoid a reporting requirement.

Both checks are **relative to the customer's own history**, never a claim about any real country
or person, which keeps them defensible: we are not ranking real jurisdictions by risk, only noting
that a payment pattern is new or unusual for that one customer — exactly the same kind of
behaviour-relative signal `llm/rag/evidence.behaviour_evidence` already produces for the anomaly
engine (it reuses the same `(transaction, history)` shape and the same counterparty/country
helpers, so this is additive, not a parallel implementation).

**Environmental and Social are NOT implemented.** A real E or S score needs, at minimum, a
counterparty or industry-sector classification (the synthetic data's customer `occupation` is a
personal job title, not a counterparty sector) and an external emissions or sustainability
dataset. Neither exists in this project. Inventing numbers for them would break this project's own
rule that no fabricated result is ever reported, so they are left as a documented extension point
instead.

## How it fits the architecture

It is a new `EvidenceSource.ESG` evidence item, `evidence_id` prefix `ESG-<transaction_id>`,
produced the same way reconciliation, anomaly and KYC evidence are. It is wired into
`guardrails/policy_checks.engine_floor`: if either signal fires, the case floor is raised to
REVIEW, through a new `PolicyConfig.esg_governance_forces_review` switch (default on) — exactly
the same policy-floor mechanism already used for the other three engines, so a governance flag
can raise a case's risk level but, like every other signal, never decide the case alone.

## What is not done yet

It is a standalone, tested module, not yet called from the live multi-agent workflow
(`agents/coordinator/workflow.py`): wiring it in means adding one more evidence item inside the
existing `analyze_anomalies` step (no new workflow state is needed, since it reuses that step's
inputs), which was judged out of scope to rush in alongside a formal state-machine change this
close to a review. The demo script proves the scorer runs correctly end to end on real synthetic
data (17,023 transactions checked, 1,257 flagged: 1,188 new cross-border counterparty, 77 possible
structuring — see `evaluation/reports/esg/governance_demo.json`); it has not been added to the
committed fine-tuning/evaluation dataset, and is not part of any reported RQ result.

## Limitations

- The reporting threshold is a configurable, illustrative figure, not tied to any real
  jurisdiction's actual law.
- No currency normalisation: the structuring check only compares transactions in the same
  currency, so it understates structuring that spans currencies.
- There is no ground-truth governance-risk label in the synthetic generator, so the demo reports
  descriptive counts only, not precision/recall or a confidence interval.
