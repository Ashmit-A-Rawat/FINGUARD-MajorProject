import { useCallback, useEffect, useMemo, useState } from "react";
import { Api } from "./api";
import { CaseView } from "./components/CaseView";
import { Inbox } from "./components/Inbox";
import { Login } from "./components/Login";
import { NewCase } from "./components/NewCase";
import { Badge, Empty } from "./components/ui";
import type { CaseSummary, Session } from "./types";

// sessionStorage: the login survives a reload of this tab only, and is gone when the tab closes.
const KEY = "fin-guard-session";
const load = (): Session | null => {
  try {
    return JSON.parse(sessionStorage.getItem(KEY) ?? "null") as Session | null;
  } catch {
    return null;
  }
};

export default function App() {
  const [session, setSession] = useState<Session | null>(load);
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [status, setStatus] = useState("");
  const [selected, setSelected] = useState<string | null>(() => decodeURIComponent(location.hash.replace(/^#\/case\//, "")) || null);
  const [creating, setCreating] = useState(false);

  const logout = useCallback(() => {
    sessionStorage.removeItem(KEY);
    setSession(null);
    setCases([]);
  }, []);
  const api = useMemo(() => (session ? new Api(session.token, logout) : null), [session, logout]);

  const refresh = useCallback(async () => {
    if (!api) return;
    try {
      setCases((await api.listCases(status || undefined)).items);
    } catch {
      /* a 401 already triggered logout; other errors keep the last list */
    }
  }, [api, status]);

  useEffect(() => { void refresh(); }, [refresh]);
  const busy = cases.some((c) => c.job_state === "queued" || c.job_state === "running");
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(() => void refresh(), 3000);
    return () => clearInterval(t);
  }, [busy, refresh]);

  const select = (id: string | null) => {
    setSelected(id);
    history.replaceState(null, "", id ? `#/case/${encodeURIComponent(id)}` : "#");
  };

  if (!session || !api) {
    return <Login onLogin={(s) => { sessionStorage.setItem(KEY, JSON.stringify(s)); setSession(s); }} />;
  }
  return (
    <div className="flex h-screen flex-col">
      <div className="flex items-center gap-3 border-b border-slate-200 bg-white px-4 py-2 text-sm">
        <span className="font-semibold">FIN-GUARD</span>
        <Badge tone="amber">synthetic data · advisory only</Badge>
        <span className="ml-auto text-slate-600">{session.username} <Badge>{session.role}</Badge></span>
        <button type="button" onClick={logout} className="rounded border border-slate-300 px-2 py-0.5">Sign out</button>
      </div>
      <div className="grid min-h-0 flex-1 grid-cols-[22rem_1fr]">
        <Inbox cases={cases} selected={selected} status={status} onStatus={setStatus} onSelect={select} onNew={() => setCreating(true)} canCreate={session.role === "analyst"} />
        <main className="min-h-0 overflow-auto p-4">
          {selected ? <CaseView key={selected} caseId={selected} api={api} session={session} onChanged={() => void refresh()} /> : <Empty>Select a case from the inbox.</Empty>}
        </main>
      </div>
      {creating && <NewCase api={api} onClose={() => setCreating(false)} onCreated={(id) => { setCreating(false); select(id); void refresh(); }} />}
    </div>
  );
}
