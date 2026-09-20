---
document_id: KB-TXN-002
title: Anomaly Triage Runbook
version: 1.3
source: FIN-GUARD project internal policy (synthetic)
effective_date: 2025-02-01
---
# Anomaly Triage Runbook

> SYNTHETIC POLICY. Invented for the FIN-GUARD research prototype. It is not real regulation or legal advice.

## Triage steps
Start from the flagged transaction and look at the customer's previous thirty days. Note which anomaly signals fired and how strong each is. Check the KYC status, then the reconciliation result for the same transactions, then look for a legitimate explanation. Write down each fact separately from any conclusion you draw from it.

## Decisions
There are three decision states. CLEAR means the activity has a credible legitimate explanation supported by evidence. REVIEW means the evidence is incomplete or mixed and another analyst or more information is needed. ESCALATE means the evidence points to possible financial crime and a senior reviewer must decide. When in doubt between CLEAR and REVIEW, choose REVIEW.

## Legitimate explanations to consider
Before escalating, consider common innocent causes: a large planned purchase, salary or bonus payments, a business paying suppliers, travel abroad, seasonal spending, and a customer changing bank details. Repeated transfers of equal amounts are often rent or loan instalments. A legitimate explanation must be supported by evidence such as a prior recurring pattern; it must not simply be assumed.

## Combining signals
A single weak signal, such as a night-time transaction, rarely justifies escalation. Several independent signals together, for example a large amount to a new beneficiary in a new country at night, raise the level of concern. Also weigh contradictions in the KYC record and unresolved reconciliation discrepancies.

## Evidence to record
Record the transaction identifiers, the anomaly signals and their values, the customer's normal range, the KYC and reconciliation findings, and the reason for the decision. Every claim in the case report must point to evidence identifiers so a reviewer can verify it.
