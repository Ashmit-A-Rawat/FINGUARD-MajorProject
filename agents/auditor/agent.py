"""Auditor agent. Responsibility: verify identity documents (KYC) and score transactions for
anomalies. Deterministic: wraps the KYC matcher and the anomaly model; never calls an LLM and never
looks at reconciliation or knowledge-base content."""

from datetime import datetime

from agents.context import AgentContext
from agents.contracts import (
    AnomalyAnalysisInput,
    AnomalyAnalysisOutput,
    KYCAnalysisInput,
    KYCAnalysisOutput,
    KYCCandidateSummary,
    KYCDocumentFinding,
)
from llm.rag.evidence import (
    anomaly_model_evidence,
    behaviour_evidence,
    customer_evidence,
    kyc_match_evidence,
    transaction_evidence,
)

TOP_N = 5


class AuditorAgent:
    name = "auditor"

    def __init__(self, ctx: AgentContext) -> None:
        self._ctx = ctx

    def analyze_kyc(self, inp: KYCAnalysisInput, now: datetime) -> KYCAnalysisOutput:
        case = inp.case
        customer_id = case.customer.customer.customer_id
        findings: list[KYCDocumentFinding] = []
        for document in case.kyc_documents:
            results = self._ctx.kyc_matcher.match(document, top_n=TOP_N)
            own = next(
                (
                    (rank, r)
                    for rank, r in enumerate(results, start=1)
                    if r.candidate_id == customer_id
                ),
                None,
            )
            findings.append(
                KYCDocumentFinding(
                    document_id=document.record.document_id,
                    own_record_rank=own[0] if own else None,
                    own_final_score=own[1].final_score if own else None,
                    own_confidence=own[1].confidence if own else None,
                    own_reasons=own[1].match_reasons if own else [],
                    own_contradictions=own[1].contradictory_evidence if own else [],
                    other_strong_candidates=[
                        KYCCandidateSummary(
                            candidate_id=r.candidate_id,
                            final_score=r.final_score,
                            is_match=r.is_match,
                            contradictions=r.contradictory_evidence,
                        )
                        for r in results
                        if r.candidate_id != customer_id and r.is_match
                    ],
                )
            )
        evidence = [
            customer_evidence(case.customer.customer, now),
            *(kyc_match_evidence(f, now) for f in findings),
        ]
        return KYCAnalysisOutput(findings=findings, evidence=evidence)

    def analyze_anomalies(self, inp: AnomalyAnalysisInput, now: datetime) -> AnomalyAnalysisOutput:
        case = inp.case
        history = self._ctx.store.transactions_by_customer[case.customer.customer.customer_id]
        findings = []
        evidence = []
        for canonical in case.focus_transactions:
            tx = canonical.transaction
            finding = self._ctx.anomaly.score(tx.transaction_id)
            findings.append(finding)
            evidence += [
                transaction_evidence(tx, now),
                behaviour_evidence(tx, history, now),
                anomaly_model_evidence(finding, now),
            ]
        return AnomalyAnalysisOutput(findings=findings, evidence=evidence)
