# Tripwire evaluation (final)

## v1 (21 attacks, 9 benign)
- regex: recall 21/21 = 1.00 [0.85, 1.00]; false positives 0/9 [0.00, 0.30]
- semantic (TRAINED ON THIS SET: not a held-out estimate): recall 21/21 = 1.00 [0.85, 1.00]; false positives 0/9 [0.00, 0.30]
- either (TRAINED ON THIS SET: not a held-out estimate): recall 21/21 = 1.00 [0.85, 1.00]; false positives 0/9 [0.00, 0.30]

## v2 (24 attacks, 17 benign)
- regex: recall 24/24 = 1.00 [0.86, 1.00]; false positives 1/17 [0.01, 0.27]
- semantic (TRAINED ON THIS SET: not a held-out estimate): recall 23/24 = 0.96 [0.80, 0.99]; false positives 0/17 [0.00, 0.18]
- either (TRAINED ON THIS SET: not a held-out estimate): recall 24/24 = 1.00 [0.86, 1.00]; false positives 1/17 [0.01, 0.27]

## v3 (30 attacks, 26 benign)
- regex: recall 7/30 = 0.23 [0.12, 0.41]; false positives 1/26 [0.01, 0.19]
- semantic (TRAINED ON THIS SET: not a held-out estimate): recall 28/30 = 0.93 [0.79, 0.98]; false positives 0/26 [0.00, 0.13]
- either (TRAINED ON THIS SET: not a held-out estimate): recall 28/30 = 0.93 [0.79, 0.98]; false positives 1/26 [0.01, 0.19]

## v4 (24 attacks, 20 benign)
- regex: recall 6/24 = 0.25 [0.12, 0.45]; false positives 0/20 [0.00, 0.16]
- semantic: recall 19/24 = 0.79 [0.60, 0.91]; false positives 0/20 [0.00, 0.16]
- either: recall 19/24 = 0.79 [0.60, 0.91]; false positives 0/20 [0.00, 0.16]

## real un-injected case evidence: 0 free-text strings in 228 cases, 0 flagged by the semantic detector
