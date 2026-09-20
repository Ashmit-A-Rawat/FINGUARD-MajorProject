"""Self-critique of an investigation, WITHOUT asking the model whether it is sure.

An LLM critiquing its own output shares its blind spots, and an injected instruction that fooled the
generator can fool the critic too. Instead the critique is mechanical: verify every claim against
the cited evidence (evidence_validator), apply deterministic policy (policy_checks), and combine the
result with the engine floor. Unsupported claims and policy violations mark the output as NOT
validated and raise the advisory decision to at least REVIEW.
"""

from collections.abc import Sequence

from backend.app.schemas.domain import Decision, Evidence
from guardrails.evidence_validator import EvidenceIndex, check_claim, check_text_grounded
from guardrails.models import ClaimCheck, CritiqueReport
from guardrails.policy_checks import DECISION_RANK, PolicyConfig, check_policies
from knowledge_base.models import RetrievedChunk
from llm.schemas import InvestigationOutput


class SelfCritique:
    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    def review(
        self,
        evidence: Sequence[Evidence],
        investigation: InvestigationOutput | None,
        chunks: Sequence[RetrievedChunk] = (),
    ) -> CritiqueReport:
        index = EvidenceIndex(evidence, chunks)
        checks: list[ClaimCheck] = []
        grounded, grounded_reason = True, "no investigation to check"
        if investigation is not None:
            for i, finding in enumerate(investigation.findings):
                checks.append(
                    check_claim(
                        i, finding.statement, finding.kind.value, finding.evidence_ids, index
                    )
                )
            for j, ref in enumerate(investigation.evidence):
                if index.resolve(ref) is None:
                    checks.append(
                        ClaimCheck(
                            index=-2 - j,
                            claim=f"evidence list entry '{ref}'",
                            kind="reference",
                            evidence_ids=[ref],
                            status="unsupported",
                            reason="cited evidence does not exist",
                        )
                    )
            grounded, grounded_reason = check_text_grounded(investigation.summary, index)

        policy = check_policies(investigation, evidence, checks, grounded, self.config)
        model_decision = investigation.recommended_action if investigation else None
        candidates = [d for d in (model_decision, policy.floor) if d is not None]
        if any(c.status == "unsupported" for c in checks):
            candidates.append(Decision.REVIEW)  # unsupported claims are routed to REVIEW
        guardrail = max(candidates, key=lambda d: DECISION_RANK[d]) if candidates else None

        adjustments: list[str] = []
        if guardrail != model_decision and guardrail is not None:
            was = model_decision.value if model_decision else "no model output"
            why = policy.floor_reasons or ["claims that could not be supported by the evidence"]
            adjustments.append(
                f"advisory decision raised from {was} to {guardrail.value}: " + "; ".join(why)
            )

        violations = [f for f in policy.flags if f.is_violation]
        unsupported = [c for c in checks if c.status == "unsupported"]
        validated = investigation is not None and not unsupported and grounded and not violations
        notes: list[str] = []
        if investigation is None:
            notes.append("No investigation was produced to validate.")
        if unsupported:
            notes.append(f"{len(unsupported)} unsupported claim(s); do not rely on them.")
        unverifiable = sum(c.status == "unverifiable" for c in checks)
        if unverifiable:
            notes.append(
                f"{unverifiable} claim(s) could not be machine-verified (no checkable content)."
            )
        if not grounded:
            notes.append(f"Summary is not grounded in the evidence: {grounded_reason}.")
        for flag in violations:
            notes.append(f"Policy violation {flag.code}: {flag.detail}")
        return CritiqueReport(
            claim_checks=checks,
            summary_grounded=grounded,
            summary_reason=grounded_reason,
            policy=policy,
            model_decision=model_decision,
            guardrail_decision=guardrail,
            adjustments=adjustments,
            validated=validated,
            notes=notes,
        )
