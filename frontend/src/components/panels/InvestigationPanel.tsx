import type { CaseDetail } from "../../types";
import { Badge, Banner, Card, DecisionBadge, Empty, SupportBadge } from "../ui";

export function InvestigationPanel({ detail, onCite }: { detail: CaseDetail; onCite: (id: string) => void }) {
  const inv = detail.investigation;
  const meta = detail.investigation_meta;
  const findings = detail.report?.model_findings ?? [];
  const cite = (raw: string) => raw.replace(/^[EK]:/, "");
  return (
    <Card id="investigation" title="AI investigation" aside={inv && <DecisionBadge decision={inv.recommended_action} label="model proposes" />}>
      <div className="space-y-3 text-sm">
        <Banner tone="amber">
          Advisory text written by a small language model. It can be wrong, and it can be manipulated by text inside the case. Rely on the engine facts and the verdicts below; a human decides.
        </Banner>
        {meta?.is_mock && <Banner tone="red">MOCK OUTPUT: no language model produced this text.</Banner>}
        {!inv ? (
          <Empty>{detail.summary.job_state === "done" ? "The model did not produce a valid investigation for this case." : "Waiting for the analysis to finish…"}</Empty>
        ) : (
          <>
            <p className="rounded bg-slate-50 p-2">{inv.summary}</p>
            <div>
              <h3 className="mb-1 font-medium">Findings</h3>
              <ul className="space-y-2">
                {inv.findings.map((f, i) => {
                  const check = findings[i];
                  return (
                    <li key={i} className="rounded border border-slate-200 p-2">
                      <div className="mb-1 flex flex-wrap items-center gap-2">
                        <Badge tone={f.kind === "fact" ? "blue" : "slate"}>{f.kind}</Badge>
                        <SupportBadge status={check?.support ?? null} />
                      </div>
                      <p>{f.statement}</p>
                      <div className="mt-1 flex flex-wrap gap-1 text-xs">
                        {f.evidence_ids.length === 0 ? <span className="text-red-700">no citation</span> : f.evidence_ids.map((id) => (
                          <button key={id} type="button" onClick={() => onCite(cite(id))} className="rounded bg-sky-50 px-1.5 py-0.5 text-sky-800 underline decoration-dotted hover:bg-sky-100">{id}</button>
                        ))}
                      </div>
                      {check?.support_reason && check.support !== "supported" && <p className="mt-1 text-xs text-slate-600">{check.support_reason}</p>}
                    </li>
                  );
                })}
              </ul>
            </div>
            {inv.uncertainties.length > 0 && <div><h3 className="font-medium">Uncertainties</h3><ul className="list-disc pl-5">{inv.uncertainties.map((u) => <li key={u}>{u}</li>)}</ul></div>}
            {inv.recommendations.length > 0 && <div><h3 className="font-medium">Suggested next steps (advisory)</h3><ul className="list-disc pl-5">{inv.recommendations.map((r) => <li key={r}>{r}</li>)}</ul></div>}
            <p className="text-xs text-slate-500">
              Model's own confidence {(inv.confidence * 100).toFixed(0)}% (uncalibrated: not a probability).{" "}
              {meta && <>Model {meta.model} · prompt {meta.prompt_version} · {meta.attempts} attempt(s) · {meta.latency_s.toFixed(1)} s</>}
            </p>
          </>
        )}
      </div>
    </Card>
  );
}
