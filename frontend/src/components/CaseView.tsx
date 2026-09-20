import { useCallback, useEffect, useState } from "react";
import type { Api } from "../api";
import type { CaseDetail, Session } from "../types";
import { AnomalyPanel } from "./panels/AnomalyPanel";
import { AuditPanel } from "./panels/AuditPanel";
import { CritiquePanel } from "./panels/CritiquePanel";
import { CustomerPanel } from "./panels/CustomerPanel";
import { DecisionPanel } from "./panels/DecisionPanel";
import { EvidencePanel } from "./panels/EvidencePanel";
import { InvestigationPanel } from "./panels/InvestigationPanel";
import { ReconciliationPanel } from "./panels/ReconciliationPanel";
import { TimelinePanel } from "./panels/TimelinePanel";
import { Badge, Banner, DecisionBadge, Empty } from "./ui";

const NAV = [["customer", "Customer"], ["timeline", "Timeline"], ["anomaly", "Anomaly"], ["reconciliation", "Reconciliation"], ["evidence", "Evidence"], ["investigation", "AI investigation"], ["critique", "Self-critique"], ["decision", "Decision"], ["audit", "Audit trail"]];

export function CaseView({ caseId, api, session, onChanged }: { caseId: string; api: Api; session: Session; onChanged: () => void }) {
  const [detail, setDetail] = useState<CaseDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [highlight, setHighlight] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setDetail(await api.getCase(caseId));
      setError(null);
    } catch {
      setError("Could not load this case.");
    }
  }, [api, caseId]);

  useEffect(() => {
    setDetail(null);
    void load();
  }, [load]);

  const running = detail && (detail.summary.job_state === "queued" || detail.summary.job_state === "running");
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => void load(), 2000);
    return () => clearInterval(timer);
  }, [running, load]);

  const cite = (id: string) => {
    setHighlight(id);
    document.getElementById(`ev-${id}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  if (error) return <Banner tone="red">{error}</Banner>;
  if (!detail) return <Empty>Loading case…</Empty>;
  const s = detail.summary;
  const warnings = detail.report?.warnings ?? [];
  const updated = (d: CaseDetail) => { setDetail(d); onChanged(); };

  return (
    <div className="space-y-4">
      <header className="sticky top-0 z-10 -mx-4 border-b border-slate-200 bg-slate-50/95 px-4 py-2 backdrop-blur">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-lg font-semibold">{s.case_id}</h1>
          <Badge>{s.status.replace("_", " ")}</Badge>
          <DecisionBadge decision={s.advisory_decision} label="advisory" />
          {s.proposed_decision && s.proposed_decision !== s.advisory_decision && <DecisionBadge decision={s.proposed_decision} label="model said" />}
          {s.is_mock && <Badge tone="red">MOCK</Badge>}
          {s.validated === false && <Badge tone="amber">not validated</Badge>}
        </div>
        <nav aria-label="Sections" className="mt-1 flex flex-wrap gap-x-3 text-xs">
          {NAV.map(([id, label]) => <a key={id} href={`#${id}`} className="text-sky-800 underline decoration-dotted">{label}</a>)}
        </nav>
      </header>

      {running && <Banner tone="blue">The analysis is running ({s.job_state}). This page updates automatically.</Banner>}
      {s.job_state === "failed" && <Banner tone="red">The analysis job failed: {s.job_error}</Banner>}
      {warnings.length > 0 && (
        <details open className="rounded border border-amber-300 bg-amber-50 p-2 text-sm text-amber-900">
          <summary className="cursor-pointer font-medium">{warnings.length} warning(s) to read before deciding</summary>
          <ul className="mt-1 list-disc pl-5">{warnings.map((w) => <li key={w}>{w}</li>)}</ul>
        </details>
      )}

      <CustomerPanel detail={detail} />
      <TimelinePanel items={detail.timeline} />
      <div className="grid gap-4 xl:grid-cols-2">
        <AnomalyPanel findings={detail.anomaly_findings} />
        <ReconciliationPanel results={detail.reconciliation_results} focusIds={s.focus_transaction_ids} />
      </div>
      <EvidencePanel detail={detail} highlight={highlight} />
      <InvestigationPanel detail={detail} onCite={cite} />
      <CritiquePanel detail={detail} />
      <DecisionPanel detail={detail} session={session} api={api} onUpdated={updated} />
      <AuditPanel key={`${s.case_id}-${s.updated_at}`} caseId={s.case_id} api={api} />
    </div>
  );
}
