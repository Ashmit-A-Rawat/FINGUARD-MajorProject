import { useState } from "react";
import type { Api } from "../../api";
import { ApiError } from "../../api";
import type { CaseDetail, Decision, Session } from "../../types";
import { decisionState } from "../decisionRules";
import { Banner, Card, DecisionBadge } from "../ui";

const CHOICES: Decision[] = ["CLEAR", "REVIEW", "ESCALATE"];

export function DecisionPanel({ detail, session, api, onUpdated }: { detail: CaseDetail; session: Session; api: Api; onUpdated: (d: CaseDetail) => void }) {
  const [choice, setChoice] = useState<Decision>("REVIEW");
  const [reason, setReason] = useState("");
  const [ack, setAck] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const state = decisionState(detail, session, choice, reason, ack);
  const id = detail.summary.case_id;

  async function run(action: () => Promise<CaseDetail>) {
    setBusy(true);
    setError(null);
    try {
      onUpdated(await action());
      setReason("");
      setAck(false);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  const so = detail.sign_off;
  return (
    <Card id="decision" title="Decision" aside={<div className="flex gap-1"><DecisionBadge decision={detail.report?.proposed_decision ?? null} label="model" /><DecisionBadge decision={detail.report?.advisory_decision ?? null} label="advisory" /></div>}>
      <div className="space-y-3 text-sm">
        <p className="text-xs text-slate-500">The system is advisory. It never freezes an account, blocks a card, contacts a customer or files a report. Your decision is recorded under your name.</p>
        {so && (
          <Banner tone="blue">
            Closed as <b>{so.decision}</b> by {so.reviewer}{so.second_reviewer ? ` (confirmed by ${so.second_reviewer})` : ""} on {so.at.replace("T", " ").slice(0, 19)}. Reason: {so.reason}
          </Banner>
        )}
        {state.note && <Banner tone={state.mode === "waiting" ? "amber" : "blue"}>{state.note}</Banner>}
        {detail.pending_escalation && (
          <Banner tone="amber">Escalation proposed by <b>{detail.pending_escalation.proposed_by}</b>: {detail.pending_escalation.reason}</Banner>
        )}

        {state.mode === "decide" && (
          <fieldset className="space-y-2">
            <legend className="font-medium">Your decision</legend>
            <div className="flex gap-2" role="radiogroup" aria-label="Decision">
              {CHOICES.map((c) => (
                <label key={c} className={`cursor-pointer rounded border px-3 py-1 ${choice === c ? "border-slate-900 bg-slate-900 text-white" : "border-slate-300"}`}>
                  <input type="radio" name="decision" value={c} checked={choice === c} onChange={() => setChoice(c)} className="sr-only" />{c}
                </label>
              ))}
            </div>
            {state.needsOverride && (
              <label className="flex items-start gap-2 rounded border border-red-300 bg-red-50 p-2 text-red-900">
                <input type="checkbox" checked={ack} onChange={(e) => setAck(e.target.checked)} className="mt-1" />
                <span>The advisory decision is <b>{detail.report?.advisory_decision}</b>. I have read the engine findings and I am overriding it. This override is recorded.</span>
              </label>
            )}
          </fieldset>
        )}

        {(state.mode === "decide" || state.mode === "confirm-escalation") && (
          <div>
            <label htmlFor="reason" className="mb-1 block font-medium">{state.mode === "confirm-escalation" ? "Your reason for confirming or rejecting" : "Reason (required)"}</label>
            <textarea id="reason" value={reason} onChange={(e) => setReason(e.target.value)} rows={3} maxLength={2000} className="w-full rounded border border-slate-300 p-2" />
          </div>
        )}

        {error && <Banner tone="red">{error}</Banner>}

        {state.mode === "decide" && (
          <button type="button" disabled={!state.canSubmit || busy} onClick={() => run(() => api.signOff(id, choice, reason, ack))}
            className="rounded bg-slate-900 px-4 py-2 text-white disabled:opacity-40">
            {choice === "ESCALATE" ? "Propose escalation" : `Sign off as ${choice}`}
          </button>
        )}
        {state.mode === "confirm-escalation" && (
          <div className="flex gap-2">
            <button type="button" disabled={!state.canSubmit || busy} onClick={() => run(() => api.confirmEscalation(id, reason))} className="rounded bg-red-700 px-4 py-2 text-white disabled:opacity-40">Confirm escalation</button>
            <button type="button" disabled={!state.canSubmit || busy} onClick={() => run(() => api.rejectEscalation(id, reason))} className="rounded border border-slate-400 px-4 py-2 disabled:opacity-40">Reject (keep in review)</button>
          </div>
        )}
      </div>
    </Card>
  );
}
