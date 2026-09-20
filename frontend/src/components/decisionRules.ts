import type { CaseDetail, Decision, Session } from "../types";

export interface DecisionState {
  mode: "read-only" | "decide" | "confirm-escalation" | "waiting";
  note: string | null;
  needsOverride: boolean; // choosing CLEAR against a REVIEW/ESCALATE advisory
  canSubmit: boolean;
}

/** UI-side mirror of the server's rules. The server is authoritative and re-checks everything. */
export function decisionState(
  detail: CaseDetail,
  session: Session,
  choice: Decision,
  reason: string,
  acknowledged: boolean,
): DecisionState {
  const status = detail.summary.status;
  const pending = detail.pending_escalation;
  if (status === "closed") return { mode: "read-only", note: "This case is closed.", needsOverride: false, canSubmit: false };
  if (session.role !== "analyst")
    return { mode: "read-only", note: "Only analysts can decide cases. Auditors and administrators have read-only access.", needsOverride: false, canSubmit: false };
  if (status !== "human_review")
    return { mode: "read-only", note: "The analysis is still running.", needsOverride: false, canSubmit: false };
  if (pending) {
    if (pending.proposed_by.toLowerCase() === session.username.toLowerCase())
      return { mode: "waiting", note: "You proposed this escalation. A different analyst must confirm it (four-eyes rule).", needsOverride: false, canSubmit: false };
    return { mode: "confirm-escalation", note: null, needsOverride: false, canSubmit: reason.trim().length > 0 };
  }
  const advisory = detail.report?.advisory_decision ?? null;
  const needsOverride = choice === "CLEAR" && (advisory === "REVIEW" || advisory === "ESCALATE");
  return {
    mode: "decide",
    note: choice === "ESCALATE" ? "ESCALATE needs a second, different analyst to confirm before the case closes." : null,
    needsOverride,
    canSubmit: reason.trim().length > 0 && (!needsOverride || acknowledged),
  };
}
