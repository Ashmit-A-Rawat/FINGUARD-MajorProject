"""Deterministic knowledge-base queries derived from case evidence (no LLM involved)."""

from backend.app.schemas.domain import Evidence, EvidenceSource

DRIVER_TOPICS = {
    "z_amount_vs_history": "high-value transactions",
    "amount_over_prior_max": "high-value transactions",
    "log_amount_usd": "high-value transactions",
    "hour_deviation": "unusual transaction time",
    "is_night": "unusual transaction time",
    "hour_sin": "unusual transaction time",
    "hour_cos": "unusual transaction time",
    "log_count_10min": "rapid bursts",
    "log_count_1h": "abnormal velocity",
    "log_count_24h": "abnormal velocity",
    "log_sum_amount_24h": "abnormal velocity",
    "is_new_counterparty_country": "geographic change",
    "is_cross_border": "geographic change",
    "is_new_counterparty": "new counterparties",
    "log_prior_same_amount_to_counterparty": "repeated identical transfers",
}
DECISION_QUERY = "decision states CLEAR REVIEW ESCALATE human sign-off"


def build_queries(evidence: list[Evidence]) -> list[str]:
    queries: list[str] = []
    for item in evidence:
        payload = item.payload
        if item.source == EvidenceSource.RECONCILIATION and "rule_id" in payload:
            queries.append(f"{payload['rule_id']} {payload.get('field', '')} reconciliation")
        elif item.evidence_id.startswith("ANOM-") and payload.get("flagged"):
            topics = [
                DRIVER_TOPICS[d[0]] for d in payload.get("top_drivers", []) if d[0] in DRIVER_TOPICS
            ]
            queries.append("anomaly triage " + " ".join(dict.fromkeys(topics)))
        elif item.evidence_id.startswith("KYCM-"):
            contradictions = payload.get("own_contradictions", [])
            if contradictions or payload.get("other_strong_candidates"):
                queries.append(
                    "KYC "
                    + " ".join(str(c).split(":")[0] for c in contradictions[:2])
                    + " duplicate namesake"
                )
    queries.append(DECISION_QUERY)
    return list(dict.fromkeys(q.strip() for q in queries))
