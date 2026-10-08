import type { ReactNode, SVGProps } from "react";
import type { Decision, SupportStatus } from "../types";

const TONES: Record<string, string> = {
  green: "bg-emerald-50 text-emerald-800 ring-emerald-300",
  amber: "bg-amber-50 text-amber-800 ring-amber-300",
  red: "bg-red-50 text-red-800 ring-red-300",
  slate: "bg-slate-100 text-slate-700 ring-slate-300",
  blue: "bg-brand-50 text-brand-700 ring-brand-300",
};

const DOTS: Record<string, string> = {
  green: "bg-emerald-500",
  amber: "bg-amber-500",
  red: "bg-red-500",
  slate: "bg-slate-400",
  blue: "bg-brand-500",
};

export function Badge({ tone = "slate", children, title }: { tone?: keyof typeof TONES; children: ReactNode; title?: string }) {
  return (
    <span title={title} className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${TONES[tone]}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${DOTS[tone]}`} aria-hidden="true" />
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

/* Small inline icon set (outline style, 1.5px stroke) so the UI needs no icon-library dependency. */
export type IconName =
  | "shield" | "user" | "clock" | "chart" | "scale" | "book" | "bot" | "check-shield"
  | "gavel" | "history" | "alert" | "logout" | "plus" | "lock" | "chevron-right" | "search";

const PATHS: Record<IconName, string> = {
  shield: "M12 3 4.5 6v5.5c0 5 3.3 8.8 7.5 10 4.2-1.2 7.5-5 7.5-10V6L12 3Z",
  user: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm-7 9a7 7 0 0 1 14 0",
  clock: "M12 7v5l3.5 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  chart: "M4 20V10m6 10V4m6 16v-7",
  scale: "M12 3v18M5 7l-3 6a3 3 0 0 0 6 0l-3-6Zm14 0-3 6a3 3 0 0 0 6 0l-3-6ZM4 7h16M9 21h6",
  book: "M4 19.5A2.5 2.5 0 0 1 6.5 17H20M4 19.5A2.5 2.5 0 0 0 6.5 22H20V3H6.5A2.5 2.5 0 0 0 4 5.5v14Z",
  bot: "M12 2v3M5 9h14a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-9a1 1 0 0 1 1-1Zm2.5 5v2m9-2v2M9 17h6",
  "check-shield": "m9 12 2 2 4-4M12 3l7.5 3v5.5c0 5-3.3 8.8-7.5 10-4.2-1.2-7.5-5-7.5-10V6L12 3Z",
  gavel: "m14 4 6 6M9.5 8.5l-6 6 3 3 6-6m-1-5 5 5M5 19l2-2M2 22l4-4",
  history: "M3 12a9 9 0 1 0 3-6.7M3 12V5m0 7h6M12 7v5l3 2",
  alert: "M12 9v4m0 4h.01M10.3 3.9 2.6 17a1.5 1.5 0 0 0 1.3 2.3h16.2a1.5 1.5 0 0 0 1.3-2.3L13.7 3.9a1.5 1.5 0 0 0-2.6 0Z",
  logout: "M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9",
  plus: "M12 5v14M5 12h14",
  lock: "M7 10V7a5 5 0 0 1 10 0v3M5 10h14v10H5V10Z",
  "chevron-right": "m9 6 6 6-6 6",
  search: "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16Zm10 2-4.3-4.3",
};

export function Icon({ name, className = "h-4 w-4" }: { name: IconName; className?: string } & SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true">
      <path d={PATHS[name]} />
    </svg>
  );
}

export function Card({ id, title, icon, children, aside }: { id: string; title: string; icon?: IconName; children: ReactNode; aside?: ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} className="scroll-mt-16 rounded-xl border border-slate-200 bg-white p-4 shadow-sm shadow-slate-200/50 transition-shadow hover:shadow-md">
      <div className="mb-3 flex items-center justify-between gap-2 border-b border-slate-100 pb-2">
        <h2 id={`${id}-title`} className="flex items-center gap-2 text-base font-semibold text-slate-800">
          {icon && (
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-50 text-brand-600" aria-hidden="true">
              <Icon name={icon} />
            </span>
          )}
          {title}
        </h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

export function Banner({ tone = "amber", children }: { tone?: "amber" | "red" | "blue"; children: ReactNode }) {
  const styles = {
    amber: "border-amber-300 bg-amber-50 text-amber-900",
    red: "border-red-300 bg-red-50 text-red-900",
    blue: "border-brand-300 bg-brand-50 text-brand-900",
  };
  const icon: IconName = tone === "red" ? "alert" : tone === "blue" ? "shield" : "alert";
  return (
    <div role="note" className={`flex items-start gap-2 rounded-lg border px-3 py-2 text-sm ${styles[tone]}`}>
      <Icon name={icon} className="mt-0.5 h-4 w-4 shrink-0" />
      <div>{children}</div>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="text-sm text-slate-500">{children}</p>;
}

export function Json({ value }: { value: unknown }) {
  return <pre className="max-h-64 overflow-auto rounded-lg bg-slate-900 p-2 text-xs text-slate-100">{JSON.stringify(value, null, 2)}</pre>;
}

export const fmtTime = (iso: string) => iso.replace("T", " ").slice(0, 19);
export const fmtMoney = (amount: number, currency: string) =>
  `${amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ${currency}`;
