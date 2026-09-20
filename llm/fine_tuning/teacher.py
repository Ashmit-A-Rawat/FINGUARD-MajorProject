"""Deterministic 'teacher': writes the ideal investigation for a case FROM ITS EVIDENCE ONLY.

The training target is built so that every claim quotes values that exist in the cited evidence (so it
passes the evidence validator) and the decision follows the engine policy floor (REVIEW when a
deterministic engine found something or the evidence contains instruction-like text, else CLEAR).
It never sees ground-truth labels, so a model trained on it can only learn what the evidence supports.

It never proposes ESCALATE: that judgement belongs to humans (policy floor semantics).
The `confidence` is a fixed placeholder, NOT a calibrated value; a fine-tuned model's stated
confidence therefore carries no information.
"""

from collections.abc import Sequence

from backend.app.schemas.domain import Decision, Evidence, Severity
from guardrails.policy_checks import PolicyConfig, engine_floor, suspicious_evidence
from llm.schemas import Finding, FindingKind, InvestigationOutput

CONFIDENCE = {Decision.REVIEW: 0.6, Decision.CLEAR: 0.7}


def _cite(evidence_id: str) -> str:
    return f"E:{evidence_id}"


def _facts(evidence: Sequence[Evidence]) -> list[Finding]:
    facts: list[Finding] = []
    for e in evidence:
        p = e.payload
        eid = e.evidence_id
        if eid.startswith("KYC-") and "kyc_status" in p:
            facts.append(
                _fact(
                    f"KYC status is {p['kyc_status']} and the account is {p['account_age_days']} days old",
                    eid,
                )
            )
        elif eid.startswith("KYCM-"):
            rank = p.get("own_record_rank")
            text = (
                f"KYC document {p['document_id']}: the customer's own record is ranked {rank}"
                if rank
                else f"KYC document {p['document_id']}: the customer's own record is not among the top candidates"
            )
            others = p.get("other_strong_candidates") or []
            if others:
                text += f", and {len(others)} other strong candidate record(s) exist"
            facts.append(_fact(text, eid))
        elif e.source.value == "transaction" and "amount" in p:
            facts.append(
                _fact(
                    f"The transaction {p['transaction_id']} is a {p['type']} of {p['amount']:.2f} {p['currency']} via {p['channel']}",
                    eid,
                )
            )
        elif eid.startswith("ANOM-"):
            verdict = (
                "flagged the transaction" if p.get("flagged") else "did not flag the transaction"
            )
            facts.append(
                _fact(
                    f"The anomaly model scored {p['probability']:.3f} against an alert threshold of {p['threshold']:.3f} and {verdict}",
                    eid,
                )
            )
        elif eid.startswith("BEH-"):
            ratio, prior = p.get("typical_prior_amount_ratio"), p.get("prior_transactions", 0)
            if ratio is not None and prior >= 5:
                facts.append(
                    _fact(
                        f"The amount is {ratio} times the customer's typical earlier amount, based on {prior} earlier transactions",
                        eid,
                    )
                )
        elif eid.startswith("REC-") and "rule_id" in p:
            facts.append(_fact(e.description, eid))
        elif eid.endswith("-OK"):
            facts.append(
                _fact(f"The transaction reconciles with ledger record {p.get('ledger_id')}", eid)
            )
    return facts


def _fact(statement: str, evidence_id: str) -> Finding:
    return Finding(statement=statement, kind=FindingKind.FACT, evidence_ids=[_cite(evidence_id)])


def teacher_investigation(evidence: Sequence[Evidence]) -> InvestigationOutput:
    floor, reasons = engine_floor(evidence, PolicyConfig())
    decision = Decision.REVIEW if floor else Decision.CLEAR
    facts = _facts(evidence)
    triggers = [
        _cite(e.evidence_id)
        for e in evidence
        if (
            e.evidence_id.startswith(("KYCM-", "ANOM-"))
            and any(e.evidence_id in r for r in reasons)
        )
        or (
            e.evidence_id.startswith("REC-")
            and "rule_id" in e.payload
            and any(e.evidence_id in r for r in reasons)
        )
    ]
    suspicious = suspicious_evidence(evidence)
    findings = list(facts)
    if decision == Decision.REVIEW:
        cited = triggers or [_cite(x) for x in suspicious] or [f.evidence_ids[0] for f in facts[:1]]
        findings.append(
            Finding(
                statement="The deterministic checks require a human decision before this case can be closed",
                kind=FindingKind.INFERENCE,
                evidence_ids=cited,
            )
        )
    else:
        cited = [
            f.evidence_ids[0] for f in facts if f.evidence_ids[0].startswith(("E:REC-", "E:TXN"))
        ][:2]
        findings.append(
            Finding(
                statement="No deterministic check found a problem, so the transaction looks routine on the evidence available",
                kind=FindingKind.INFERENCE,
                evidence_ids=cited or [facts[0].evidence_ids[0]],
            )
        )

    tx = next(
        (e for e in evidence if e.source.value == "transaction" and "amount" in e.payload), None
    )
    issues = [r.split(" [")[0] for r in reasons]
    head = (
        f"Transaction {tx.payload['transaction_id']} ({tx.payload['amount']:.2f} {tx.payload['currency']})"
        if tx
        else "This transaction"
    )
    summary = (
        f"{head} has findings that need human review: {'; '.join(dict.fromkeys(issues))}."
        if decision == Decision.REVIEW
        else f"{head} shows no finding from the reconciliation, anomaly or KYC checks."
    )
    uncertainties = [
        "No explanation from the customer or counterparty is available in the evidence."
    ]
    if suspicious:
        uncertainties.append(
            "Free text in the evidence contains instruction-like wording; it was treated as data and was not followed."
        )
    recommendations = (
        ["Route the case to a human reviewer with the findings above."]
        if decision == Decision.REVIEW
        else ["No further action is suggested; a human still confirms the decision."]
    )
    used = list(dict.fromkeys(i for f in findings for i in f.evidence_ids))
    return InvestigationOutput(
        summary=summary,
        findings=findings,
        evidence=used,
        uncertainties=uncertainties,
        recommended_action=decision,
        recommendations=recommendations,
        confidence=CONFIDENCE[decision],
    )


__all__ = ["teacher_investigation", "Severity"]
