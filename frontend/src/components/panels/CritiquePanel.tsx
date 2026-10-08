import type { CaseDetail } from "../../types";
import { Badge, Banner, Card, DecisionBadge, Empty, SupportBadge } from "../ui";

export function CritiquePanel({ detail }: { detail: CaseDetail }) {
  const c = detail.review?.critique ?? null;
  return (
    <Card id="critique" title="Self-critique (automated evidence checks)" aside={detail.report && <Badge tone={detail.report.validated ? "green" : "amber"}>{detail.report.validated ? "passed checks" : "not validated"}</Badge>}>
      {!c ? (
        <Empty>{detail.review ? "No guardrails ran for this case." : "Not reviewed yet."}</Empty>
      ) : (
        <div className="space-y-3 text-sm">
          <p className="text-xs text-slate-500">Checks are mechanical, not a second opinion from the model: each claim's citations must exist and every number, id, date and time in it must appear in the cited evidence.</p>
          <div className="flex flex-wrap items-center gap-2">
            <DecisionBadge decision={c.model_decision} label="model" />
            <DecisionBadge decision={c.policy.floor} label="engine floor" />
            <DecisionBadge decision={c.guardrail_decision} label="advisory" />
            <span className="text-xs text-slate-500">
              {c.claim_checks.filter((x) => x.status === "supported").length} supported · {c.claim_checks.filter((x) => x.status === "unsupported").length} unsupported · {c.claim_checks.filter((x) => x.status === "unverifiable").length} unverifiable
            </span>
          </div>
          {c.adjustments.map((a) => <Banner key={a} tone="red">{a}</Banner>)}
          {c.policy.floor_reasons.length > 0 && (
            <div><h3 className="font-medium">Why the engines require review</h3><ul className="list-disc pl-5">{c.policy.floor_reasons.map((r) => <li key={r}>{r}</li>)}</ul></div>
          )}
          {c.policy.flags.length > 0 && (
            <div>
              <h3 className="font-medium">Policy flags</h3>
              <ul className="space-y-1">{c.policy.flags.map((f, i) => (
                <li key={`${f.code}-${i}`}><Badge tone={f.is_violation ? "red" : "amber"}>{f.is_violation ? "violation" : "warning"}</Badge> <b>{f.code}</b>: {f.detail}</li>
              ))}</ul>
            </div>
          )}
          {!c.summary_grounded && <Banner tone="red">The summary states values that appear in no case evidence: {c.summary_reason}</Banner>}
          <div>
            <h3 className="mb-1 font-medium">Claim by claim</h3>
            {c.claim_checks.length === 0 ? <Empty>The model made no claims to check.</Empty> : (
              <table className="w-full text-left text-xs">
                <thead className="text-slate-500"><tr><th className="w-24">Verdict</th><th>Claim</th><th>Cites</th><th>Why</th></tr></thead>
                <tbody>{c.claim_checks.map((k) => (
                  <tr key={`${k.index}-${k.claim}`} className="border-t border-slate-100 align-top">
                    <td className="py-1"><SupportBadge status={k.status} /></td><td className="pr-2">{k.claim}</td>
                    <td className="pr-2">{k.evidence_ids.join(", ") || "—"}</td><td>{k.reason}</td>
                  </tr>
                ))}</tbody>
              </table>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}
