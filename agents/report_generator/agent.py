"""Report agent. Responsibility: assemble the explainable case report. Deterministic; no LLM.

Keeps three things visibly apart: engine facts (deterministic), model statements (unvalidated),
and warnings. The model's decision is only ever a PROPOSAL.
"""

from agents.contracts import CaseReport, ReportInput, ReportItem
from backend.app.schemas.domain import Decision, Severity
from llm.schemas import normalize_citation

ENGINE_VERSIONS = {
    "kyc": "hybrid_structured (BM25 + dense + DOB/address verification)",
    "anomaly": "xgboost",
    "reconciliation": "rules REC-001..REC-010",
}


def _engine_facts(inp: ReportInput) -> list[ReportItem]:
    facts: list[ReportItem] = []
    for k in inp.kyc_findings:
        rank = (
            f"ranked {k.own_record_rank}" if k.own_record_rank else "NOT among the top candidates"
        )
        score = (
            f" (score {k.own_final_score:.2f}, confidence {k.own_confidence})"
            if k.own_final_score
            else ""
        )
        text = f"KYC document {k.document_id}: the customer's own record is {rank}{score}."
        if k.own_contradictions:
            text += f" Contradictory evidence: {'; '.join(k.own_contradictions)}."
        if k.other_strong_candidates:
            ids = ", ".join(c.candidate_id for c in k.other_strong_candidates)
            text += f" Other strong candidate record(s): {ids}."
        facts.append(ReportItem(text=text, evidence_ids=[f"KYCM-{k.document_id}"], origin="engine"))
    for a in inp.anomaly_findings:
        drivers = ", ".join(f"{name} ({value:+.2f})" for name, value in a.top_drivers)
        facts.append(
            ReportItem(
                text=(
                    f"Anomaly model score {a.probability:.3f} vs threshold {a.threshold:.3f}: "
                    f"{'FLAGGED' if a.flagged else 'not flagged'}. Top drivers: {drivers}."
                ),
                evidence_ids=[f"ANOM-{a.transaction_id}"],
                origin="engine",
            )
        )
    focus = set(inp.focus_transaction_ids)
    for r in inp.reconciliation_results:
        if r.transaction_id not in focus:
            continue
        if not r.discrepancies:
            facts.append(
                ReportItem(
                    text=f"Transaction {r.transaction_id} reconciles with the ledger.",
                    evidence_ids=[f"REC-{r.transaction_id}-OK"],
                    origin="engine",
                )
            )
        for d in r.discrepancies:
            facts.append(
                ReportItem(
                    text=(
                        f"{d.rule_id} ({d.severity.value}) on {d.field}: expected {d.expected}, "
                        f"actual {d.actual}. {d.explanation}"
                    ),
                    evidence_ids=[f"REC-{r.transaction_id}-{d.rule_id}"],
                    origin="engine",
                )
            )
    return facts


def _engine_concerns(inp: ReportInput) -> list[str]:
    concerns = [
        f"{d.rule_id}"
        for r in inp.reconciliation_results
        if r.transaction_id in set(inp.focus_transaction_ids)
        for d in r.discrepancies
        if d.severity == Severity.HIGH
    ]
    concerns += [f"anomaly flag on {a.transaction_id}" for a in inp.anomaly_findings if a.flagged]
    concerns += [
        f"KYC document {k.document_id}"
        for k in inp.kyc_findings
        if k.other_strong_candidates or k.own_record_rank != 1
    ]
    return concerns


class ReportAgent:
    name = "report_generator"

    def generate(self, inp: ReportInput) -> CaseReport:
        inv, meta, warnings = inp.investigation, inp.investigation_meta, []
        critique = inp.review.critique if inp.review else None
        is_mock = bool(meta and meta.is_mock)
        validated = bool(inp.review and inp.review.validated) and not is_mock
        if is_mock:
            warnings.append("MOCK OUTPUT: no language model produced this investigation.")
        if not validated:
            warnings.append(
                "UNVALIDATED: the investigation has not passed automated evidence checks"
                + (" (mock output is never validated)." if is_mock else ".")
            )
        if inp.failed_step:
            warnings.append(f"Workflow step '{inp.failed_step}' failed; this report is incomplete.")
        if inv is None:
            warnings.append("No AI investigation is available; rely on the engine facts.")
        if any(a.in_training_period for a in inp.anomaly_findings):
            warnings.append(
                "Anomaly score is optimistic: the transaction is in the model's training period."
            )
        if meta and meta.excluded_suspicious_chunks:
            warnings.append(
                "Reference text withheld from the model as suspicious: "
                + ", ".join(meta.excluded_suspicious_chunks)
            )
        if critique is None:  # no guardrails ran: keep the cheap consistency check
            concerns = _engine_concerns(inp)
            if inv is not None and inv.recommended_action == Decision.CLEAR and concerns:
                warnings.append(
                    "CONSISTENCY WARNING: the model proposes CLEAR but the engines report: "
                    + "; ".join(concerns)
                    + ". (No guardrails ran to enforce a minimum.)"
                )
        checks = {c.index: c for c in critique.claim_checks} if critique else {}
        model_findings = [
            ReportItem(
                text=f.statement,
                origin="model",
                evidence_ids=[normalize_citation(i) for i in f.evidence_ids],
                support=checks[i].status if i in checks else None,
                support_reason=checks[i].reason if i in checks else None,
            )
            for i, f in enumerate(inv.findings if inv else [])
        ]
        if critique is not None:
            counts = critique.counts
            if counts["unsupported"]:
                warnings.append(
                    f"{counts['unsupported']} model claim(s) are UNSUPPORTED by the cited "
                    "evidence; do not rely on them."
                )
            warnings += [f"POLICY {f.code}: {f.detail}" for f in critique.policy.flags]
            warnings += critique.adjustments
        advisory = (
            critique.guardrail_decision if critique else (inv.recommended_action if inv else None)
        )
        return CaseReport(
            case_id=inp.case_id,
            customer_id=inp.customer_id,
            focus_transaction_ids=inp.focus_transaction_ids,
            generated_at=inp.generated_at,
            proposed_decision=inv.recommended_action if inv else None,
            proposed_decision_note="The model's own proposal; a human reviewer decides.",
            advisory_decision=advisory,
            decision_adjustments=list(critique.adjustments) if critique else [],
            claim_summary=critique.counts if critique else {},
            policy_flags=[f.code for f in critique.policy.flags] if critique else [],
            validated=validated,
            engine_facts=_engine_facts(inp),
            model_findings=model_findings,
            recommendations=inv.recommendations if inv else [],
            uncertainties=inv.uncertainties if inv else [],
            warnings=warnings,
            evidence_index={e.evidence_id: e.description for e in inp.evidence},
            provenance={
                "prompt_version": meta.prompt_version if meta else None,
                "provider": meta.provider if meta else None,
                "model": meta.model if meta else None,
                "is_mock": meta.is_mock if meta else None,
                "attempts": meta.attempts if meta else None,
                "knowledge_chunks": [c.chunk_id for c in inp.knowledge_chunks],
                "engines": ENGINE_VERSIONS,
                "model_stated_confidence": inv.confidence if inv else None,
                "guardrails": "self_critique+policy_floor" if critique else "none",
            },
        )
