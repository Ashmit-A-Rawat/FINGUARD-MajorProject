---
document_id: KB-KYC-003
title: Entity Resolution Guidance
version: 1.1
source: FIN-GUARD project internal policy (synthetic)
effective_date: 2025-02-01
---
# Entity Resolution Guidance

> SYNTHETIC POLICY. Invented for the FIN-GUARD research prototype. It is not real regulation or legal advice.

## Reading match scores
The matcher returns several scores for each candidate. The lexical score reflects word overlap in names, the semantic score reflects similarity of meaning learned by an embedding model, and the reranker score is a learned estimate that two names refer to the same person. The structured match score measures agreement of date of birth and address. The final score combines the name score and the structured score. Read every component; the final score alone hides which evidence agreed.

## Contradictory evidence
Every disagreement between the document and the candidate is listed as contradictory evidence, for example a date-of-birth mismatch or a different city. A high name score together with contradictory evidence on date of birth and address is the typical signature of a namesake, not a match. Do not confirm a match while important contradictions remain unexplained.

## Confidence bands
A high band means the final score is well above the decision threshold, medium means it is just above it, and low means it is below it. Medium confidence always requires human review. High confidence still requires sign-off before any record is merged or changed. Low confidence means the candidate should not be treated as the same person.

## When automated matching is not enough
Escalate to manual review when the candidates have similar scores, when the best candidate has contradictory date-of-birth evidence, when the name is very common, or when the document shows signs of alteration. The matcher is a decision aid, never a decision-maker.
