import type { ReconciliationResult } from "../../types";
import { Badge, Card, Empty } from "../ui";

const SEVERITY = { high: "red", medium: "amber", low: "slate" } as const;

export function ReconciliationPanel({ results, focusIds }: { results: ReconciliationResult[]; focusIds: string[] }) {
  const focus = results.filter((r) => focusIds.includes(r.transaction_id));
  const context = results.filter((r) => !focusIds.includes(r.transaction_id));
  const contextIssues = context.filter((r) => r.status === "discrepancy").length;
  return (
    <Card id="reconciliation" title="Reconciliation results">
      {focus.length === 0 ? (
        <Empty>No reconciliation results yet.</Empty>
      ) : (
        <div className="space-y-3 text-sm">
          {focus.map((r) => (
            <div key={r.transaction_id}>
              <div className="flex items-center gap-2">
                <span className="font-medium">{r.transaction_id}</span>
                <Badge tone={r.status === "reconciled" ? "green" : "red"}>{r.status}</Badge>
                <span className="text-slate-500">ledger {r.ledger_id ?? "none"}</span>
              </div>
              {r.discrepancies.length > 0 && (
                <table className="mt-2 w-full text-left text-xs">
                  <thead className="text-slate-500"><tr><th>Rule</th><th>Field</th><th>Expected</th><th>Actual</th><th>Difference</th><th>Severity</th></tr></thead>
                  <tbody>
                    {r.discrepancies.map((d) => (
                      <tr key={d.rule_id} className="border-t border-slate-100 align-top">
                        <td className="py-1 pr-2 font-medium">{d.rule_id}</td><td className="pr-2">{d.field}</td>
                        <td className="pr-2">{d.expected ?? "—"}</td><td className="pr-2">{d.actual ?? "—"}</td>
                        <td className="pr-2">{d.difference ?? "—"}</td>
                        <td><Badge tone={SEVERITY[d.severity]}>{d.severity}</Badge></td>
                      </tr>
                    ))}
                    {r.discrepancies.map((d) => <tr key={`${d.rule_id}-x`}><td colSpan={6} className="pb-2 text-slate-600">{d.rule_id}: {d.explanation}</td></tr>)}
                  </tbody>
                </table>
              )}
            </div>
          ))}
          <p className="text-xs text-slate-500">{contextIssues} of {context.length} recent transactions of this customer also have reconciliation findings. Status rules (REC-009/010) are observations, not integrity faults.</p>
        </div>
      )}
    </Card>
  );
}
