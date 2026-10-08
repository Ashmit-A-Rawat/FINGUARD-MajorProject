import type { TimelineItem } from "../../types";
import { Badge, Card, Empty, fmtMoney, fmtTime } from "../ui";

function AmountStrip({ items }: { items: TimelineItem[] }) {
  if (items.length < 2) return null;
  const logs = items.map((i) => Math.log10(Math.max(i.amount, 1)));
  const lo = Math.min(...logs), hi = Math.max(...logs), span = hi - lo || 1;
  const w = 600, h = 60, gap = w / items.length;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Transaction amounts over time (log scale)" className="mb-3 h-16 w-full">
      {items.map((it, i) => {
        const barH = 6 + ((logs[i] - lo) / span) * (h - 10);
        const fill = it.is_focus ? "#dc2626" : it.reconciliation === "discrepancy" ? "#d97706" : "#94a3b8";
        return <rect key={it.transaction_id} x={i * gap + 1} y={h - barH} width={Math.max(gap - 2, 2)} height={barH} fill={fill}><title>{`${fmtTime(it.timestamp)} ${fmtMoney(it.amount, it.currency)}`}</title></rect>;
      })}
    </svg>
  );
}

export function TimelinePanel({ items }: { items: TimelineItem[] }) {
  return (
    <Card id="timeline" title="Transaction timeline" aside={<span className="text-xs text-slate-500">focus in red, reconciliation findings in amber</span>}>
      {items.length === 0 ? (
        <Empty>No transactions to show yet.</Empty>
      ) : (
        <>
          <AmountStrip items={items} />
          <div className="max-h-72 overflow-auto">
            <table className="w-full text-left text-sm">
              <thead className="sticky top-0 bg-white text-xs text-slate-500"><tr><th className="py-1">When</th><th>Amount</th><th>Type</th><th>Channel</th><th>Route</th><th>Flags</th></tr></thead>
              <tbody>
                {[...items].reverse().map((t) => (
                  <tr key={t.transaction_id} className={`border-t border-slate-100 ${t.is_focus ? "bg-red-50 font-medium" : ""}`}>
                    <td className="py-1 pr-2 whitespace-nowrap">{fmtTime(t.timestamp)}</td>
                    <td className="pr-2 whitespace-nowrap">{fmtMoney(t.amount, t.currency)}</td>
                    <td className="pr-2">{t.type}</td>
                    <td className="pr-2">{t.channel}</td>
                    <td className="pr-2">{t.sender_country}→{t.receiver_country}</td>
                    <td className="space-x-1">
                      {t.is_focus && <Badge tone="red">focus</Badge>}
                      {t.reconciliation === "discrepancy" && <Badge tone="amber" title={t.reconciliation_rules.join(", ")}>recon</Badge>}
                      {t.anomaly_flagged && <Badge tone="red">anomaly</Badge>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Card>
  );
}
