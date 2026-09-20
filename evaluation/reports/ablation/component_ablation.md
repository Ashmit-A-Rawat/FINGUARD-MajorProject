# EXP-ABL-01: engine-floor component ablation

SYNTHETIC. Flag = engine floor is REVIEW.


## heldout_eval (n=24)

| configuration | recall: reconciliation | recall: behavioural | false-flag rate: clean | precision |
|---|---|---|---|---|
| full (all engines, default policy) | 1.00 [1.00, 1.00] n=8 | 0.38 [0.12, 0.75] n=8 | 0.00 [0.00, 0.00] n=8 | 1.00 |
| reconciliation only | 1.00 [1.00, 1.00] n=8 | 0.00 [0.00, 0.00] n=8 | 0.00 [0.00, 0.00] n=8 | 1.00 |
| anomaly model only | 0.00 [0.00, 0.00] n=8 | 0.25 [0.00, 0.62] n=8 | 0.00 [0.00, 0.00] n=8 | 1.00 |
| KYC matcher only | 0.00 [0.00, 0.00] n=8 | 0.12 [0.00, 0.38] n=8 | 0.00 [0.00, 0.00] n=8 | 1.00 |
| reconciliation + anomaly | 1.00 [1.00, 1.00] n=8 | 0.25 [0.00, 0.62] n=8 | 0.00 [0.00, 0.00] n=8 | 1.00 |
| reconciliation + KYC | 1.00 [1.00, 1.00] n=8 | 0.12 [0.00, 0.38] n=8 | 0.00 [0.00, 0.00] n=8 | 1.00 |
| anomaly + KYC | 0.00 [0.00, 0.00] n=8 | 0.38 [0.12, 0.75] n=8 | 0.00 [0.00, 0.00] n=8 | 1.00 |
| full, LOW-severity findings also count | 1.00 [1.00, 1.00] n=8 | 0.38 [0.12, 0.75] n=8 | 0.00 [0.00, 0.00] n=8 | 1.00 |
| full, only HIGH-severity findings count | 0.62 [0.25, 0.88] n=8 | 0.38 [0.12, 0.75] n=8 | 0.00 [0.00, 0.00] n=8 | 1.00 |
| full, settlement-status rules also count | 1.00 [1.00, 1.00] n=8 | 0.38 [0.12, 0.75] n=8 | 0.00 [0.00, 0.00] n=8 | 1.00 |

## pooled_uninjected (n=228)

| configuration | recall: reconciliation | recall: behavioural | false-flag rate: clean | precision |
|---|---|---|---|---|
| full (all engines, default policy) | 1.00 [1.00, 1.00] n=67 | 0.64 [0.51, 0.75] n=53 | 0.06 [0.02, 0.10] n=108 | 0.94 |
| reconciliation only | 1.00 [1.00, 1.00] n=67 | 0.00 [0.00, 0.00] n=53 | 0.00 [0.00, 0.00] n=108 | 1.00 |
| anomaly model only | 0.04 [0.00, 0.10] n=67 | 0.60 [0.47, 0.72] n=53 | 0.00 [0.00, 0.00] n=108 | 1.00 |
| KYC matcher only | 0.04 [0.00, 0.10] n=67 | 0.06 [0.00, 0.13] n=53 | 0.06 [0.02, 0.10] n=108 | 0.50 |
| reconciliation + anomaly | 1.00 [1.00, 1.00] n=67 | 0.60 [0.47, 0.72] n=53 | 0.00 [0.00, 0.00] n=108 | 1.00 |
| reconciliation + KYC | 1.00 [1.00, 1.00] n=67 | 0.06 [0.00, 0.13] n=53 | 0.06 [0.02, 0.10] n=108 | 0.92 |
| anomaly + KYC | 0.09 [0.03, 0.16] n=67 | 0.64 [0.51, 0.75] n=53 | 0.06 [0.02, 0.10] n=108 | 0.87 |
| full, LOW-severity findings also count | 1.00 [1.00, 1.00] n=67 | 0.64 [0.51, 0.75] n=53 | 0.06 [0.02, 0.10] n=108 | 0.94 |
| full, only HIGH-severity findings count | 0.81 [0.70, 0.90] n=67 | 0.64 [0.51, 0.75] n=53 | 0.06 [0.02, 0.10] n=108 | 0.94 |
| full, settlement-status rules also count | 1.00 [1.00, 1.00] n=67 | 0.64 [0.51, 0.75] n=53 | 0.06 [0.02, 0.10] n=108 | 0.94 |
