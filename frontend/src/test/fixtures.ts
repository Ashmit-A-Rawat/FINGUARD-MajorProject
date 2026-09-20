import type { CaseDetail, CaseSummary, Session } from "../types";

export const analyst: Session = { token: "t", username: "alice", role: "analyst" };
export const auditor: Session = { token: "t", username: "audrey", role: "auditor" };

export const summary = (over: Partial<CaseSummary> = {}): CaseSummary => ({
  case_id: "CASE-CUST-1-TXN-1", customer_id: "CUST-1", focus_transaction_ids: ["TXN-1"], status: "human_review",
  job_state: "done", job_error: null, advisory_decision: "REVIEW", proposed_decision: "CLEAR", validated: false,
  warnings: 2, is_mock: false, created_by: "alice", created_at: "2025-07-01T09:00:00", updated_at: "2025-07-01T09:01:00",
  pending_escalation_by: null, ...over,
});

export function detail(over: Partial<CaseDetail> = {}, sum: Partial<CaseSummary> = {}): CaseDetail {
  return {
    summary: summary(sum),
    customer: { customer_id: "CUST-1", kyc_status: "verified", account_type: "personal", account_age_days: 400, country: "US", documents: [{ document_id: "DOC-1", document_type: "passport", issue_date: "2022-01-01", expiry_date: "2027-01-01" }] },
    timeline: [{ transaction_id: "TXN-1", timestamp: "2025-07-01T03:00:00", amount: 900, currency: "USD", type: "transfer_out", channel: "web", sender_country: "US", receiver_country: "BR", is_focus: true, reconciliation: "discrepancy", reconciliation_rules: ["REC-003"], anomaly_probability: 0.9, anomaly_flagged: true }],
    anomaly_findings: [{ transaction_id: "TXN-1", probability: 0.93, flagged: true, threshold: 0.41, top_drivers: [["z_amount_vs_history", 1.2], ["hour_deviation", -0.4]], in_training_period: false, model: "xgboost" }],
    kyc_findings: [{ document_id: "DOC-1", own_record_rank: 1, own_final_score: 0.9, own_confidence: "high", own_reasons: [], own_contradictions: [], other_strong_candidates: [] }],
    reconciliation_results: [{ transaction_id: "TXN-1", ledger_id: "LED-1", status: "discrepancy", discrepancies: [{ field: "amount", expected: "900.00", actual: "1012.50", difference: "+112.50 (+12.5%)", severity: "high", rule_id: "REC-003", explanation: "Ledger posted 1012.50." }] }],
    evidence: [{ evidence_id: "TXN-1", source: "transaction", description: "transfer of 900.00 USD", payload: { amount: 900 }, created_at: "2025-07-01T00:00:00" }],
    knowledge_chunks: [], retrieval_queries: [],
    investigation: { summary: "A 900.00 USD transfer.", findings: [
      { statement: "The transfer was 900.00 USD", kind: "fact", evidence_ids: ["E:TXN-1"] },
      { statement: "The transfer was 950.00 USD", kind: "fact", evidence_ids: ["E:TXN-1"] },
      { statement: "Looks unusual", kind: "inference", evidence_ids: [] },
    ], evidence: ["TXN-1"], uncertainties: ["Counterparty unknown"], recommended_action: "CLEAR", recommendations: ["Ask the customer"], confidence: 0.9 },
    investigation_meta: { prompt_version: "investigation-v1", provider: "qwen", model: "qwen2.5", is_mock: false, attempts: 1, latency_s: 12.3 },
    review: { validated: false, notes: [], critique: {
      claim_checks: [
        { index: 0, claim: "The transfer was 900.00 USD", kind: "fact", evidence_ids: ["E:TXN-1"], status: "supported", reason: "all good" },
        { index: 1, claim: "The transfer was 950.00 USD", kind: "fact", evidence_ids: ["E:TXN-1"], status: "unsupported", reason: "not found in the cited evidence: numbers 950" },
        { index: 2, claim: "Looks unusual", kind: "inference", evidence_ids: [], status: "unsupported", reason: "no evidence is cited" },
      ],
      summary_grounded: true, summary_reason: "grounded",
      policy: { flags: [{ code: "CLEAR_BELOW_ENGINE_FLOOR", detail: "the model proposes CLEAR but engines found REC-003", is_violation: true }], floor: "REVIEW", floor_reasons: ["reconciliation REC-003 (high)"] },
      model_decision: "CLEAR", guardrail_decision: "REVIEW", adjustments: ["advisory decision raised from CLEAR to REVIEW: reconciliation REC-003 (high)"], validated: false, notes: [] } },
    report: { case_id: "CASE-CUST-1-TXN-1", proposed_decision: "CLEAR", advisory_decision: "REVIEW", decision_adjustments: [], claim_summary: { supported: 1, unsupported: 2, unverifiable: 0 }, policy_flags: [], validated: false, engine_facts: [],
      model_findings: [
        { text: "The transfer was 900.00 USD", evidence_ids: ["TXN-1"], origin: "model", support: "supported", support_reason: "all good" },
        { text: "The transfer was 950.00 USD", evidence_ids: ["TXN-1"], origin: "model", support: "unsupported", support_reason: "not found in the cited evidence: numbers 950" },
        { text: "Looks unusual", evidence_ids: [], origin: "model", support: "unsupported", support_reason: "no evidence is cited" },
      ], recommendations: [], uncertainties: [], warnings: ["UNVALIDATED"], evidence_index: {}, provenance: {} },
    sign_off: null, pending_escalation: null, ...over,
  };
}
