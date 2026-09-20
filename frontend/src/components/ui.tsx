import type { ReactNode } from "react";
import type { Decision, SupportStatus } from "../types";

const TONES: Record<string, string> = {
  green: "bg-emerald-100 text-emerald-900 ring-emerald-300",
  amber: "bg-amber-100 text-amber-900 ring-amber-300",
  red: "bg-red-100 text-red-900 ring-red-300",
  slate: "bg-slate-100 text-slate-700 ring-slate-300",
  blue: "bg-sky-100 text-sky-900 ring-sky-300",
};

export function Badge({ tone = "slate", children, title }: { tone?: keyof typeof TONES; children: ReactNode; title?: string }) {
  return (
    <span title={title} className={`inline-flex items-center rounded px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${TONES[tone]}`}>
      {children}
    </span>
  );
}

const DECISION_TONE: Record<Decision, keyof typeof TONES> = { CLEAR: "green", REVIEW: "amber", ESCALATE: "red" };

export function DecisionBadge({ decision, label }: { decision: Decision | null; label?: string }) {
  if (!decision) return <Badge>{label ? `${label}: none` : "none"}</Badge>;
  return <Badge tone={DECISION_TONE[decision]}>{label ? `${label}: ${decision}` : decision}</Badge>;
}

const SUPPORT_TONE: Record<SupportStatus, keyof typeof TONES> = { supported: "green", unsupported: "red", unverifiable: "amber" };

export function SupportBadge({ status }: { status: SupportStatus | null }) {
  if (!status) return <Badge title="No guardrail ran">not checked</Badge>;
  const hint = {
    supported: "Every checkable value appears in the cited evidence",
    unsupported: "A citation or a quoted value is not backed by the cited evidence",
    unverifiable: "Nothing machine-checkable in this claim; not confirmed",
  }[status];
  return <Badge tone={SUPPORT_TONE[status]} title={hint}>{status}</Badge>;
}

export function Card({ id, title, children, aside }: { id: string; title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} className="scroll-mt-16 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 id={`${id}-title`} className="text-base font-semibold">{title}</h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

export function Banner({ tone = "amber", children }: { tone?: "amber" | "red" | "blue"; children: ReactNode }) {
  const styles = { amber: "border-amber-300 bg-amber-50 text-amber-900", red: "border-red-300 bg-red-50 text-red-900", blue: "border-sky-300 bg-sky-50 text-sky-900" };
  return <div role="note" className={`rounded border px-3 py-2 text-sm ${styles[tone]}`}>{children}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="text-sm text-slate-500">{children}</p>;
}

export function Json({ value }: { value: unknown }) {
  return <pre className="max-h-64 overflow-auto rounded bg-slate-50 p-2 text-xs">{JSON.stringify(value, null, 2)}</pre>;
}

export const fmtTime = (iso: string) => iso.replace("T", " ").slice(0, 19);
export const fmtMoney = (amount: number, currency: string) =>
  `${amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ${currency}`;
