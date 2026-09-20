// Mirrors the API's JSON (backend/app/schemas/api.py). Kept by hand; the backend is the source of truth.

export type Role = "analyst" | "auditor" | "admin";
export type Decision = "CLEAR" | "REVIEW" | "ESCALATE";
export type SupportStatus = "supported" | "unsupported" | "unverifiable";

export interface Session {
  token: string;
  username: string;
  role: Role;
}

export interface CaseSummary {
  case_id: string;
  customer_id: string;
  focus_transaction_ids: string[];
  status: string;
  job_state: "queued" | "running" | "done" | "failed";
  job_error: string | null;
  advisory_decision: Decision | null;
  proposed_decision: Decision | null;
  validated: boolean | null;
  warnings: number;
  is_mock: boolean | null;
  created_by: string;
  created_at: string;
  updated_at: string;
  pending_escalation_by: string | null;
}

export interface Evidence {
  evidence_id: string;
  source: string;
  description: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface Chunk {
  chunk_id: string;
  document_id: string;
  text: string;
  score: number;
  rank: number;
  metadata: {
    document_id: string;
    source: string;
    section: string;
    version: string;
    trust: string;
    injection_flags: string[];
    page: number | null;
  };
}

export interface Discrepancy {
  field: string;
  expected: string | null;
  actual: string | null;
  difference: string | null;
  severity: "low" | "medium" | "high";
  rule_id: string;
  explanation: string;
}

export interface ReconciliationResult {
  transaction_id: string;
  ledger_id: string | null;
  status: "reconciled" | "discrepancy";
  discrepancies: Discrepancy[];
}

export interface AnomalyFinding {
  transaction_id: string;
  probability: number;
  flagged: boolean;
  threshold: number;
  top_drivers: [string, number][];
  in_training_period: boolean;
  model: string;
}

export interface KYCFinding {
  document_id: string;
  own_record_rank: number | null;
  own_final_score: number | null;
  own_confidence: string | null;
  own_reasons: string[];
  own_contradictions: string[];
  other_strong_candidates: { candidate_id: string; final_score: number; is_match: boolean; contradictions: string[] }[];
}

export interface Finding {
  statement: string;
  kind: "fact" | "inference";
  evidence_ids: string[];
}

export interface Investigation {
  summary: string;
  findings: Finding[];
  evidence: string[];
  uncertainties: string[];
  recommended_action: Decision;
  recommendations: string[];
  confidence: number;
}

export interface ClaimCheck {
  index: number;
  claim: string;
  kind: string;
  evidence_ids: string[];
  status: SupportStatus;
  reason: string;
}

export interface PolicyFlag {
  code: string;
  detail: string;
  is_violation: boolean;
}

export interface Critique {
  claim_checks: ClaimCheck[];
  summary_grounded: boolean;
  summary_reason: string;
  policy: { flags: PolicyFlag[]; floor: Decision | null; floor_reasons: string[] };
  model_decision: Decision | null;
  guardrail_decision: Decision | null;
  adjustments: string[];
  validated: boolean;
  notes: string[];
}

export interface ReportItem {
  text: string;
  evidence_ids: string[];
  origin: "engine" | "model";
  support: SupportStatus | null;
  support_reason: string | null;
}

export interface Report {
  case_id: string;
  proposed_decision: Decision | null;
  advisory_decision: Decision | null;
  decision_adjustments: string[];
  claim_summary: Record<string, number>;
  policy_flags: string[];
  validated: boolean;
  engine_facts: ReportItem[];
  model_findings: ReportItem[];
  recommendations: string[];
  uncertainties: string[];
  warnings: string[];
  evidence_index: Record<string, string>;
  provenance: Record<string, unknown>;
}

export interface TimelineItem {
  transaction_id: string;
  timestamp: string;
  amount: number;
  currency: string;
  type: string;
  channel: string;
  sender_country: string;
  receiver_country: string;
  is_focus: boolean;
  reconciliation: "reconciled" | "discrepancy" | null;
  reconciliation_rules: string[];
  anomaly_probability: number | null;
  anomaly_flagged: boolean | null;
}

export interface CustomerPanelData {
  customer_id?: string;
  kyc_status?: string;
  account_type?: string;
  account_age_days?: number;
  country?: string;
  documents?: { document_id: string; document_type: string; issue_date: string; expiry_date: string }[];
  details?: {
    name: string;
    alternate_names: string[];
    date_of_birth: string;
    address: string;
    occupation: string;
    document_names: Record<string, string>;
    document_addresses: Record<string, string>;
    document_dates_of_birth: Record<string, string>;
  };
}

export interface CaseDetail {
  summary: CaseSummary;
  customer: CustomerPanelData;
  timeline: TimelineItem[];
  anomaly_findings: AnomalyFinding[];
  kyc_findings: KYCFinding[];
  reconciliation_results: ReconciliationResult[];
  evidence: Evidence[];
  knowledge_chunks: Chunk[];
  retrieval_queries: string[];
  investigation: Investigation | null;
  investigation_meta: { prompt_version: string; provider: string; model: string; is_mock: boolean; attempts: number; latency_s: number } | null;
  review: { validated: boolean; notes: string[]; critique: Critique | null } | null;
  report: Report | null;
  sign_off: { reviewer: string; second_reviewer: string | null; decision: Decision; reason: string; at: string } | null;
  pending_escalation: { proposed_by: string; reason: string } | null;
}

export interface AuditEvent {
  event_id: number;
  timestamp: string;
  actor: string;
  action: string;
  from_status: string | null;
  to_status: string | null;
  duration_ms: number;
  ok: boolean;
  error: string | null;
  details: Record<string, unknown>;
  prev_hash: string;
  hash: string;
}

export interface AuditResponse {
  events: AuditEvent[];
  chain: { ok: boolean; events: number; first_bad_seq: number | null; reason: string };
}

export interface Candidate {
  customer_id: string;
  transaction_id: string;
  timestamp: string;
  amount: number;
  currency: string;
  has_case: boolean;
}
