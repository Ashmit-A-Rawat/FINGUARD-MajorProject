import type { CaseDetail } from "../../types";
import { Badge, Banner, Card, Empty } from "../ui";

export function CustomerPanel({ detail }: { detail: CaseDetail }) {
  const c = detail.customer;
  const details = c.details;
  const today = detail.timeline.length ? detail.timeline[detail.timeline.length - 1].timestamp.slice(0, 10) : "";
  return (
    <Card id="customer" title="Customer & KYC">
      {!c.customer_id ? (
        <Empty>Customer data is not available yet.</Empty>
      ) : (
        <div className="space-y-3 text-sm">
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-4">
            <div><dt className="text-slate-500">Customer</dt><dd>{c.customer_id}</dd></div>
            <div><dt className="text-slate-500">KYC status</dt><dd><Badge tone={c.kyc_status === "verified" ? "green" : "amber"}>{c.kyc_status}</Badge></dd></div>
            <div><dt className="text-slate-500">Account</dt><dd>{c.account_type}, {c.account_age_days} days</dd></div>
            <div><dt className="text-slate-500">Country</dt><dd>{c.country}</dd></div>
          </dl>

          {details ? (
            <div>
              <Banner tone="blue">Personal details are shown to analysts only and every view is recorded in the audit trail.</Banner>
              <dl className="mt-2 grid grid-cols-1 gap-1 sm:grid-cols-2">
                <div><dt className="text-slate-500">Name (master record)</dt><dd>{details.name}</dd></div>
                <div><dt className="text-slate-500">Date of birth</dt><dd>{details.date_of_birth}</dd></div>
                <div className="sm:col-span-2"><dt className="text-slate-500">Address</dt><dd>{details.address}</dd></div>
              </dl>
            </div>
          ) : (
            <Empty>Personal details are hidden for your role.</Empty>
          )}

          <h3 className="pt-1 font-medium">Identity documents</h3>
          {(c.documents ?? []).map((d) => {
            const finding = detail.kyc_findings.find((k) => k.document_id === d.document_id);
            const expired = today !== "" && d.expiry_date < today;
            return (
              <div key={d.document_id} className="rounded border border-slate-200 p-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{d.document_id}</span>
                  <Badge>{d.document_type}</Badge>
                  <span className="text-slate-500">expires {d.expiry_date}</span>
                  {expired && <Badge tone="red">expired</Badge>}
                </div>
                {details && (
                  <p className="mt-1 text-xs text-slate-600">
                    On document: {details.document_names[d.document_id]} · {details.document_dates_of_birth[d.document_id]} · {details.document_addresses[d.document_id]}
                  </p>
                )}
                {finding ? (
                  <div className="mt-2 space-y-1">
                    <p>
                      Own record ranked <b>{finding.own_record_rank ?? "not in top candidates"}</b>
                      {finding.own_final_score != null && <> · score {finding.own_final_score.toFixed(2)} ({finding.own_confidence})</>}
                    </p>
                    {finding.own_contradictions.length > 0 && (
                      <ul className="list-disc pl-5 text-red-800">{finding.own_contradictions.map((x) => <li key={x}>{x}</li>)}</ul>
                    )}
                    {finding.other_strong_candidates.length > 0 && (
                      <Banner tone="amber">
                        Other strong candidate record(s): {finding.other_strong_candidates.map((o) => `${o.candidate_id} (${o.final_score.toFixed(2)})`).join(", ")}. Possible duplicate or namesake; do not merge on name alone.
                      </Banner>
                    )}
                  </div>
                ) : (
                  <Empty>No KYC analysis for this document.</Empty>
                )}
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}
