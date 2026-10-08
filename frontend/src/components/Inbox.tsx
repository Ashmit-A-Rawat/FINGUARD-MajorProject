import type { CaseSummary } from "../types";
import { Badge, DecisionBadge, Empty, Icon, fmtTime } from "./ui";

const STATUS_OPTIONS = [["", "All"], ["human_review", "Awaiting review"], ["closed", "Closed"], ["case_created", "Processing"]];

const ACCENT: Record<string, string> = { CLEAR: "border-l-emerald-400", REVIEW: "border-l-amber-400", ESCALATE: "border-l-red-400" };

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
        <h2 className="flex items-center gap-1.5 text-base font-semibold text-slate-800">
          <Icon name="book" className="h-4 w-4 text-brand-600" />Case inbox
        </h2>
        <select aria-label="Filter by status" value={status} onChange={(e) => onStatus(e.target.value)} className="ml-auto rounded-md border border-slate-300 px-1.5 py-1 text-xs">
          {STATUS_OPTIONS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
        {canCreate && (
          <button type="button" onClick={onNew} className="flex items-center gap-1 rounded-md bg-brand-600 px-2.5 py-1.5 text-sm text-white transition-colors hover:bg-brand-700">
            <Icon name="plus" className="h-3.5 w-3.5" />New case
          </button>
        )}
      </div>
      <ul className="flex-1 overflow-auto">
        {cases.length === 0 && <li className="p-3"><Empty>No cases{status ? " with this status" : " yet"}.</Empty></li>}
        {cases.map((c) => (
          <li key={c.case_id}>
            <button type="button" onClick={() => onSelect(c.case_id)} aria-current={selected === c.case_id}
              className={`w-full border-b border-l-4 border-slate-100 p-3 text-left transition-colors hover:bg-slate-50 ${ACCENT[c.advisory_decision ?? ""] ?? "border-l-transparent"} ${selected === c.case_id ? "bg-brand-50" : ""}`}>
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-sm font-medium text-slate-800">{c.customer_id}</span>
                <DecisionBadge decision={c.advisory_decision} />
              </div>
              <div className="truncate text-xs text-slate-500">{c.focus_transaction_ids.join(", ")}</div>
              <div className="mt-1.5 flex flex-wrap items-center gap-1 text-xs text-slate-500">
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
