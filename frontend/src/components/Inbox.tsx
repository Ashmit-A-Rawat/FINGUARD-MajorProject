import type { CaseSummary } from "../types";
import { Badge, DecisionBadge, Empty, fmtTime } from "./ui";

const STATUS_OPTIONS = [["", "All"], ["human_review", "Awaiting review"], ["closed", "Closed"], ["case_created", "Processing"]];

export function Inbox({ cases, selected, status, onStatus, onSelect, onNew, canCreate }: {
  cases: CaseSummary[];
  selected: string | null;
  status: string;
  onStatus: (s: string) => void;
  onSelect: (id: string) => void;
  onNew: () => void;
  canCreate: boolean;
}) {
  return (
    <aside aria-label="Case inbox" className="flex h-full flex-col border-r border-slate-200 bg-white">
      <div className="flex items-center gap-2 border-b border-slate-200 p-3">
        <h2 className="text-base font-semibold">Case inbox</h2>
        <select aria-label="Filter by status" value={status} onChange={(e) => onStatus(e.target.value)} className="ml-auto rounded border border-slate-300 px-1 py-0.5 text-sm">
          {STATUS_OPTIONS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
        {canCreate && <button type="button" onClick={onNew} className="rounded bg-slate-900 px-2 py-1 text-sm text-white">New case</button>}
      </div>
      <ul className="flex-1 overflow-auto">
        {cases.length === 0 && <li className="p-3"><Empty>No cases{status ? " with this status" : " yet"}.</Empty></li>}
        {cases.map((c) => (
          <li key={c.case_id}>
            <button type="button" onClick={() => onSelect(c.case_id)} aria-current={selected === c.case_id}
              className={`w-full border-b border-slate-100 p-3 text-left hover:bg-slate-50 ${selected === c.case_id ? "bg-sky-50" : ""}`}>
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-sm font-medium">{c.customer_id}</span>
                <DecisionBadge decision={c.advisory_decision} />
              </div>
              <div className="truncate text-xs text-slate-500">{c.focus_transaction_ids.join(", ")}</div>
              <div className="mt-1 flex flex-wrap items-center gap-1 text-xs text-slate-500">
                <Badge tone={c.status === "closed" ? "slate" : c.job_state === "failed" ? "red" : "blue"}>{c.job_state === "done" ? c.status.replace("_", " ") : c.job_state}</Badge>
                {c.is_mock && <Badge tone="red">mock</Badge>}
                {c.validated === false && <Badge tone="amber">unvalidated</Badge>}
                {c.pending_escalation_by && <Badge tone="red">escalation pending</Badge>}
                {c.warnings > 0 && <span>{c.warnings} warning(s)</span>}
                <span className="ml-auto">{fmtTime(c.created_at)}</span>
              </div>
            </button>
          </li>
        ))}
      </ul>
    </aside>
  );
}
