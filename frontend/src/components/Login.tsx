import { useState } from "react";
import { Api, ApiError } from "../api";
import type { Session } from "../types";
import { Banner } from "./ui";

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
    <main className="mx-auto mt-24 max-w-sm rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
      <h1 className="mb-1 text-xl font-semibold">FIN-GUARD</h1>
      <p className="mb-4 text-sm text-slate-600">Case review. Research prototype on synthetic data; advisory output only.</p>
      <form onSubmit={submit} className="space-y-3">
        <div><label htmlFor="username" className="mb-1 block text-sm font-medium">Username</label>
          <input id="username" autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} className="w-full rounded border border-slate-300 p-2" /></div>
        <div><label htmlFor="password" className="mb-1 block text-sm font-medium">Password</label>
          <input id="password" type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} className="w-full rounded border border-slate-300 p-2" /></div>
        {error && <Banner tone="red">{error}</Banner>}
        <button type="submit" disabled={busy || !username || !password} className="w-full rounded bg-slate-900 p-2 text-white disabled:opacity-40">Sign in</button>
      </form>
    </main>
  );
}
