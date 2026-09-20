import type { AnomalyFinding } from "../../types";
import { Badge, Banner, Card, Empty } from "../ui";

export function AnomalyPanel({ findings }: { findings: AnomalyFinding[] }) {
  return (
    <Card id="anomaly" title="Anomaly analysis">
      {findings.length === 0 ? (
        <Empty>No anomaly analysis yet.</Empty>
      ) : (
        findings.map((f) => {
          const maxDriver = Math.max(...f.top_drivers.map(([, v]) => Math.abs(v)), 0.001);
          return (
            <div key={f.transaction_id} className="space-y-2 text-sm">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium">{f.transaction_id}</span>
                <Badge tone={f.flagged ? "red" : "green"}>{f.flagged ? "flagged" : "not flagged"}</Badge>
                <span className="text-slate-500">model: {f.model}</span>
              </div>
              <div aria-label={`score ${f.probability.toFixed(3)} against threshold ${f.threshold.toFixed(3)}`}>
                <div className="relative h-4 rounded bg-slate-200">
                  <div className={`h-4 rounded ${f.flagged ? "bg-red-500" : "bg-emerald-500"}`} style={{ width: `${Math.min(f.probability, 1) * 100}%` }} />
                  <div className="absolute top-0 h-4 w-0.5 bg-slate-900" style={{ left: `${Math.min(f.threshold, 1) * 100}%` }} title="alert threshold" />
                </div>
                <p className="mt-1 text-xs text-slate-600">score {f.probability.toFixed(3)} · threshold {f.threshold.toFixed(3)} (black line)</p>
              </div>
              <div>
                <p className="text-xs font-medium text-slate-500">What drove the score</p>
                {f.top_drivers.map(([name, value]) => (
                  <div key={name} className="flex items-center gap-2 text-xs">
                    <span className="w-56 truncate">{name}</span>
                    <div className="h-2 flex-1 rounded bg-slate-100">
                      <div className={`h-2 rounded ${value >= 0 ? "bg-red-400" : "bg-sky-400"}`} style={{ width: `${(Math.abs(value) / maxDriver) * 100}%` }} />
                    </div>
                    <span className="w-14 text-right tabular-nums">{value >= 0 ? "+" : ""}{value.toFixed(2)}</span>
                  </div>
                ))}
              </div>
              {f.in_training_period && <Banner tone="amber">This transaction is inside the model's training period, so the score is optimistic.</Banner>}
            </div>
          );
        })
      )}
    </Card>
  );
}
