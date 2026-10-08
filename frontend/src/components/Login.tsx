import { useState } from "react";
import { Api, ApiError } from "../api";
import type { Session } from "../types";
import { Banner, Icon } from "./ui";

export function Login({ onLogin }: { onLogin: (s: Session) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onLogin(await Api.login(username, password));
    } catch (err) {
      setError(err instanceof ApiError && err.status === 429 ? "Too many failed attempts. Try again later." : "Invalid credentials.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-gradient-to-br from-brand-900 via-brand-700 to-brand-500 p-4">
      <div className="w-full max-w-sm rounded-2xl border border-white/10 bg-white p-7 shadow-2xl shadow-brand-900/30">
        <div className="mb-5 flex items-center gap-3">
          <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-brand-600 text-white"><Icon name="shield" className="h-6 w-6" /></span>
          <div>
            <h1 className="text-xl font-semibold text-slate-900">FIN-GUARD</h1>
            <p className="text-xs text-slate-500">Case review · research prototype</p>
          </div>
        </div>
        <p className="mb-5 text-sm text-slate-600">All data is synthetic. Output is advisory only.</p>
        <form onSubmit={submit} className="space-y-3">
          <div>
            <label htmlFor="username" className="mb-1 block text-sm font-medium text-slate-700">Username</label>
            <input id="username" autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)}
              className="w-full rounded-lg border border-slate-300 p-2 text-sm transition-colors focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100" />
          </div>
          <div>
            <label htmlFor="password" className="mb-1 block text-sm font-medium text-slate-700">Password</label>
            <input id="password" type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-lg border border-slate-300 p-2 text-sm transition-colors focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100" />
          </div>
          {error && <Banner tone="red">{error}</Banner>}
          <button type="submit" disabled={busy || !username || !password}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-brand-600 p-2.5 text-sm font-medium text-white transition-colors hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-40">
            <Icon name="lock" className="h-4 w-4" />Sign in
          </button>
        </form>
      </div>
    </main>
  );
}
