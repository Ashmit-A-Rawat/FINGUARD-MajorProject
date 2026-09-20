---
document_id: KB-REC-001
title: Reconciliation Runbook
version: 1.4
source: FIN-GUARD project internal policy (synthetic)
effective_date: 2025-02-15
---
# Reconciliation Runbook

> SYNTHETIC POLICY. Invented for the FIN-GUARD research prototype. It is not real regulation or legal advice.

## Overview
Reconciliation compares each transaction record with its ledger record and reports either reconciled or discrepancy. Each discrepancy has a rule identifier, the field involved, the expected and actual values, the difference and a severity. Source records are never modified by reconciliation; corrections are made through the normal correction process after review.

## Missing ledger entry (REC-001)
Rule REC-001 fires when a transaction has no ledger entry. Severity is high because the transaction may not have been posted at all. Check whether the posting is simply delayed, whether the transaction was cancelled, and whether the ledger feed had an outage. Do not assume the customer was charged or credited.

## Duplicate posting (REC-002)
Rule REC-002 fires when a transaction has more than one ledger entry, which risks charging or paying the customer twice. Severity is high. Identify the earliest posting as the original and treat the others as candidate duplicates. A reversal of the duplicate must be approved by a person; never reverse automatically.

## Amount mismatch (REC-003)
Rule REC-003 fires when the posted amount differs from the transaction amount by more than one cent. Severity depends on the relative difference: low below one percent, medium from one percent, and high from ten percent of the transaction amount. Check for fees, currency rounding and partial settlement before treating the difference as an error.

## Currency mismatch (REC-004)
Rule REC-004 fires when the ledger currency differs from the transaction currency. Severity is high because the amounts are then not comparable and the customer may have been converted at an unintended rate. Find out where the conversion happened.

## Reference problems (REC-005 and REC-006)
Rule REC-005 fires when a reference identifier is present on only one side, which is medium severity because the records cannot be tied by reference. Rule REC-006 fires when both sides have a reference but the values differ, which is high severity because it suggests the ledger record may belong to another transaction.

## Posting time problems (REC-007 and REC-008)
Rule REC-007 fires when the ledger posting is timestamped before the transaction occurred, which is high severity and usually indicates a clock or data error. Rule REC-008 fires when the posting is later than the permitted lag after the transaction. It is medium severity, and high when the delay exceeds three times the permitted lag.

## Settlement status (REC-009 and REC-010)
Rule REC-009 records a settlement that the ledger marks as failed. Rule REC-010 records a settlement still pending long after the transaction. Both are medium severity. These are status observations rather than data-integrity faults, and they should be handled separately from the integrity rules.
