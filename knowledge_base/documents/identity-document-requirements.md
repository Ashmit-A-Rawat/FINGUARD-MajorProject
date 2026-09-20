---
document_id: KB-KYC-002
title: Identity Document Requirements
version: 1.0
source: FIN-GUARD project internal policy (synthetic)
effective_date: 2025-01-01
---
# Identity Document Requirements

> SYNTHETIC POLICY. Invented for the FIN-GUARD research prototype. It is not real regulation or legal advice.

## Accepted documents
Three document types are accepted for identity verification: a passport, a national identity card and a driving licence. Each record must show the holder's full name, date of birth, a document number, an issue date and an expiry date. Photocopies are not accepted for initial onboarding.

## Document expiry
An expired document cannot verify identity. When the expiry date is earlier than the review date, the customer's KYC status must be set to expired and a renewed document requested. Documents that expire within thirty days should be flagged so the customer can be contacted early. A document whose expiry date is not later than its issue date is invalid and must be rejected as a data error.

## Document number handling
Document numbers are compared after normalisation: letters are upper-cased and spaces and punctuation are removed. Two different customers presenting the same document number is a serious anomaly and requires ESCALATE. A document number must never be written to logs in full; log only the last four characters.
