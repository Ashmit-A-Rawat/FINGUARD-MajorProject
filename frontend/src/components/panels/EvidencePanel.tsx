import type { CaseDetail } from "../../types";
import { Badge, Card, Empty, Json } from "../ui";

export function EvidencePanel({ detail, highlight }: { detail: CaseDetail; highlight: string | null }) {
  return (
    <Card id="evidence" title="Retrieved evidence" aside={<span className="text-xs text-slate-500">{detail.evidence.length} case items · {detail.knowledge_chunks.length} reference passages</span>}>
      {detail.evidence.length === 0 ? (
        <Empty>No evidence yet.</Empty>
      ) : (
        <div className="space-y-4 text-sm">
          <div>
            <h3 className="mb-1 font-medium">Case evidence (from the deterministic engines)</h3>
            <ul className="space-y-1">
              {detail.evidence.map((e) => (
                <li key={e.evidence_id} id={`ev-${e.evidence_id}`} className={`rounded border p-2 ${highlight === e.evidence_id ? "border-sky-500 bg-sky-50" : "border-slate-200"}`}>
                  <div className="flex flex-wrap items-center gap-2">
                    <code className="text-xs">{e.evidence_id}</code><Badge>{e.source}</Badge>
                  </div>
                  <p>{e.description}</p>
                  <details className="mt-1"><summary className="cursor-pointer text-xs text-slate-500">data</summary><Json value={e.payload} /></details>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="mb-1 font-medium">Reference passages (retrieved, treated as data)</h3>
            {detail.retrieval_queries.length > 0 && <p className="mb-1 text-xs text-slate-500">Queries: {detail.retrieval_queries.join(" · ")}</p>}
            {detail.knowledge_chunks.length === 0 ? <Empty>None retrieved.</Empty> : (
              <ul className="space-y-1">
                {detail.knowledge_chunks.map((c) => (
                  <li key={c.chunk_id} id={`ev-${c.chunk_id}`} className={`rounded border p-2 ${highlight === c.chunk_id ? "border-sky-500 bg-sky-50" : "border-slate-200"}`}>
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      <code>{c.chunk_id}</code>
                      <Badge tone={c.metadata.trust === "trusted" ? "green" : "red"}>{c.metadata.trust}</Badge>
                      <span className="text-slate-500">{c.metadata.source} · {c.metadata.section} · v{c.metadata.version} · score {c.score.toFixed(3)}</span>
                      {c.metadata.injection_flags.length > 0 && <Badge tone="red">instruction-like text: {c.metadata.injection_flags.join(", ")}</Badge>}
                    </div>
                    <p className="mt-1 text-slate-700">{c.text}</p>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}
