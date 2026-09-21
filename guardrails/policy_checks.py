"""Deterministic policy checks. The model is advisory: nothing it says can lower risk below what
the deterministic engines found.

ENGINE FLOOR: findings from reconciliation, the anomaly model and the KYC matcher imply a minimum
decision (REVIEW). The guardrail decision is max(model decision, floor). The floor never forces
ESCALATE: that judgement belongs to humans.

EVIDENCE TRIPWIRE: free text supplied by third parties (a payment memo, for example) is scanned for
instruction-like content with the same heuristic used for the knowledge base. A hit forces REVIEW
and is shown to the reviewer. It is a tripwire, not a defence: a rephrased attack can slip past it,
in which case the engine floor is what protects a case that has real problems.
"""

from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel

from backend.app.schemas.domain import Decision, Evidence, EvidenceSource, Severity
from guardrails.models import ClaimCheck, PolicyFlag, PolicyResult
from knowledge_base.ingestion.sanitize import injection_flags
from knowledge_base.ingestion.semantic_tripwire import SemanticTripwire
from llm.schemas import InvestigationOutput

DECISION_RANK = {Decision.CLEAR: 0, Decision.REVIEW: 1, Decision.ESCALATE: 2}
SEVERITY_RANK = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2}
STATUS_RULES = {"REC-009", "REC-010"}  # settlement-status observations, not integrity faults


class PolicyConfig(BaseModel):
    floor_min_severity: Severity = Severity.MEDIUM
    status_rules_force_review: bool = False
    anomaly_flag_forces_review: bool = True
    kyc_concerns_force_review: bool = True
    suspicious_evidence_forces_review: bool = True
    confidence_ceiling_with_unsupported: float = 0.5


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


FREE_TEXT_KEYS = ("memo", "note", "comment", "remark", "narrative", "text")


def free_text(payload: Any) -> Iterable[str]:
    """Strings under payload keys that hold third-party free text (the semantic tripwire reads only
    these: ids, dates and engine-generated descriptions are not free text)."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str) and any(k in str(key).lower() for k in FREE_TEXT_KEYS):
                yield value
            elif isinstance(value, (dict, list, tuple)):
                yield from free_text(value)
    elif isinstance(payload, (list, tuple)):
        for v in payload:
            yield from free_text(v)


def suspicious_evidence(
    evidence: Sequence[Evidence], semantic: SemanticTripwire | None = None
) -> list[str]:
    """Ids of evidence items whose free text looks like an instruction to the model."""
    found = []
    for item in evidence:
        texts = [item.description, *_strings(item.payload)]
        if any(injection_flags(t) for t in texts) or (
            semantic is not None and any(semantic.is_injection(t) for t in free_text(item.payload))
        ):
            found.append(item.evidence_id)
    return found


def engine_floor(
    evidence: Sequence[Evidence],
    config: PolicyConfig,
    semantic: SemanticTripwire | None = None,
) -> tuple[Decision | None, list[str]]:
    reasons: list[str] = []
    minimum = SEVERITY_RANK[config.floor_min_severity]
    for e in evidence:
        payload = e.payload
        if e.source == EvidenceSource.RECONCILIATION and "rule_id" in payload:
            rule = str(payload["rule_id"])
            if rule in STATUS_RULES and not config.status_rules_force_review:
                continue
            severity = Severity(payload.get("severity", "low"))
            if SEVERITY_RANK[severity] >= minimum:
                reasons.append(f"reconciliation {rule} ({severity.value}) [{e.evidence_id}]")
        elif e.evidence_id.startswith("ANOM-") and config.anomaly_flag_forces_review:
            if payload.get("flagged"):
                reasons.append(f"anomaly model flagged the transaction [{e.evidence_id}]")
        elif e.evidence_id.startswith("KYCM-") and config.kyc_concerns_force_review:
            if payload.get("other_strong_candidates"):
                reasons.append(f"KYC: other strong candidate record(s) [{e.evidence_id}]")
            if payload.get("own_record_rank") != 1:
                reasons.append(
                    f"KYC: the customer's own record is not the best match [{e.evidence_id}]"
                )
            if any(str(c).startswith("dob_") for c in payload.get("own_contradictions", [])):
                reasons.append(f"KYC: date-of-birth conflict [{e.evidence_id}]")
    if config.suspicious_evidence_forces_review:
        for evidence_id in suspicious_evidence(evidence, semantic):
            reasons.append(f"instruction-like text inside evidence [{evidence_id}]")
    return (Decision.REVIEW if reasons else None), reasons


def check_policies(
    investigation: InvestigationOutput | None,
    evidence: Sequence[Evidence],
    claim_checks: Sequence[ClaimCheck],
    summary_grounded: bool,
    config: PolicyConfig,
    semantic: SemanticTripwire | None = None,
) -> PolicyResult:
    floor, reasons = engine_floor(evidence, config, semantic)
    flags: list[PolicyFlag] = []
    unsupported = [c for c in claim_checks if c.status == "unsupported"]

    for evidence_id in suspicious_evidence(evidence, semantic):
        flags.append(
            PolicyFlag(
                code="SUSPICIOUS_EVIDENCE_TEXT",
                is_violation=False,
                detail=f"{evidence_id} contains instruction-like text supplied by a third party",
            )
        )
    if investigation is None:
        return PolicyResult(flags=flags, floor=floor, floor_reasons=reasons)

    if investigation.recommended_action == Decision.CLEAR:
        if floor is not None:
            flags.append(
                PolicyFlag(
                    code="CLEAR_BELOW_ENGINE_FLOOR",
                    is_violation=True,
                    detail="the model proposes CLEAR but deterministic findings require review: "
                    + "; ".join(reasons),
                )
            )
        if unsupported:
            flags.append(
                PolicyFlag(
                    code="CLEAR_WITH_UNSUPPORTED_CLAIMS",
                    is_violation=True,
                    detail=f"{len(unsupported)} claim(s) behind a CLEAR proposal are unsupported",
                )
            )
        if not any(c.status == "supported" for c in claim_checks):
            flags.append(
                PolicyFlag(
                    code="CLEAR_WITHOUT_SUPPORTED_EVIDENCE",
                    is_violation=True,
                    detail="a CLEAR proposal needs at least one verified, evidence-backed claim",
                )
            )
    if not summary_grounded:
        flags.append(
            PolicyFlag(
                code="SUMMARY_NOT_GROUNDED",
                is_violation=True,
                detail="the summary states values that appear in no case evidence",
            )
        )
    if unsupported and investigation.confidence > config.confidence_ceiling_with_unsupported:
        flags.append(
            PolicyFlag(
                code="HIGH_CONFIDENCE_WITH_UNSUPPORTED_CLAIMS",
                is_violation=False,
                detail=f"the model states confidence {investigation.confidence:.2f} "
                f"despite {len(unsupported)} unsupported claim(s)",
            )
        )
    model_text = " ".join(
        [
            investigation.summary,
            *investigation.uncertainties,
            *investigation.recommendations,
            *(f.statement for f in investigation.findings),
        ]
    )
    if injection_flags(model_text):
        flags.append(
            PolicyFlag(
                code="OUTPUT_CONTAINS_INSTRUCTION_LIKE_TEXT",
                is_violation=False,
                detail="the model output repeats instruction-like text; check if it was injected",
            )
        )
    return PolicyResult(flags=flags, floor=floor, floor_reasons=reasons)
