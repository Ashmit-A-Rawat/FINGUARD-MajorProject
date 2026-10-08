import { useState } from "react";
import type { Api } from "../../api";
import type { AuditResponse } from "../../types";
import { Badge, Banner, Card, Empty, fmtTime } from "../ui";

export function AuditPanel({ caseId, api }: { caseId: string; api: Api }) {
  const [data, setData] = useState<AuditResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      setData(await api.audit(caseId));
    } catch {
      setError("Could not load the audit trail.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card id="audit" title="Audit trail" aside={data && <Badge tone={data.chain.ok ? "green" : "red"}>{data.chain.ok ? `chain intact (${data.chain.events} events)` : "CHAIN BROKEN"}</Badge>}>
      <div className="space-y-2 text-sm">
        <p className="text-xs text-slate-500">Append-only and hash-chained: altering, deleting or reordering an event breaks every later hash. Opening the trail is itself recorded.</p>
        <button type="button" onClick={load} disabled={loading} className="rounded border border-slate-400 px-3 py-1 disabled:opacity-40">{data ? "Reload" : "Load audit trail"}</button>
        {error && <Banner tone="red">{error}</Banner>}
        {data && !data.chain.ok && <Banner tone="red">Integrity check failed at event {data.chain.first_bad_seq}: {data.chain.reason}. Treat this trail as untrustworthy and escalate to an administrator.</Banner>}
        {data && (data.events.length === 0 ? <Empty>No events.</Empty> : (
          <div className="max-h-80 overflow-auto">
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-white text-slate-500"><tr><th>#</th><th>Time</th><th>Actor</th><th>Action</th><th>Transition</th><th>ms</th><th>Hash</th></tr></thead>
              <tbody>
                {data.events.map((e) => (
                  <tr key={e.event_id} className={`border-t border-slate-100 ${e.ok ? "" : "bg-red-50"}`}>
                    <td className="py-1 pr-2">{e.event_id}</td><td className="pr-2 whitespace-nowrap">{fmtTime(e.timestamp)}</td>
                    <td className="pr-2">{e.actor}</td><td className="pr-2">{e.action}{e.error ? ` — ${e.error}` : ""}</td>
                    <td className="pr-2 whitespace-nowrap">{e.from_status ?? "·"} → {e.to_status ?? "·"}</td>
                    <td className="pr-2 tabular-nums">{e.duration_ms.toFixed(0)}</td><td><code title={e.hash}>{e.hash.slice(0, 8)}</code></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </Card>
  );
}
