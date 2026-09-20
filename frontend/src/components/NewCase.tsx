import { useEffect, useState } from "react";
import type { Api } from "../api";
import { ApiError } from "../api";
import type { Candidate } from "../types";
import { Banner, Empty, fmtMoney, fmtTime } from "./ui";

export function NewCase({ api, onClose, onCreated }: { api: Api; onClose: () => void; onCreated: (caseId: string) => void }) {
  const [items, setItems] = useState<Candidate[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    api.candidates().then(setItems).catch((e) => setError(e instanceof ApiError ? e.message : "Could not load candidates"));
  }, [api]);

  async function open(c: Candidate) {
    setBusy(c.transaction_id);
    try {
      onCreated((await api.createCase(c.customer_id, [c.transaction_id])).case_id);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not create the case");
      setBusy(null);
    }
  }

  return (
    <div role="dialog" aria-modal="true" aria-label="Open a new case" className="fixed inset-0 z-20 flex items-start justify-center bg-slate-900/40 p-6">
      <div className="max-h-[85vh] w-full max-w-3xl overflow-auto rounded-lg bg-white p-4 shadow-xl">
        <div className="mb-2 flex items-center justify-between"><h2 className="text-lg font-semibold">Open a case</h2><button type="button" onClick={onClose} className="rounded border border-slate-300 px-2 py-1 text-sm">Close</button></div>
        <p className="mb-2 text-sm text-slate-600">Recent transactions from the held-out period (after the anomaly model's training data). Analysis takes from seconds up to a minute.</p>
        {error && <Banner tone="red">{error}</Banner>}
        {!items && !error && <Empty>Loading…</Empty>}
        {items && (
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-slate-500"><tr><th>When</th><th>Customer</th><th>Transaction</th><th>Amount</th><th /></tr></thead>
            <tbody>{items.map((c) => (
              <tr key={c.transaction_id} className="border-t border-slate-100">
                <td className="py-1 pr-2 whitespace-nowrap">{fmtTime(c.timestamp)}</td><td className="pr-2">{c.customer_id}</td><td className="pr-2">{c.transaction_id}</td>
                <td className="pr-2 whitespace-nowrap">{fmtMoney(c.amount, c.currency)}</td>
                <td>{c.has_case ? <span className="text-xs text-slate-500">case exists</span> : <button type="button" disabled={busy !== null} onClick={() => open(c)} className="rounded bg-slate-900 px-2 py-0.5 text-xs text-white disabled:opacity-40">{busy === c.transaction_id ? "Opening…" : "Open"}</button>}</td>
              </tr>
            ))}</tbody>
          </table>
        )}
      </div>
    </div>
  );
}
