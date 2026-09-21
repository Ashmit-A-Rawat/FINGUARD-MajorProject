# EXP-ADV-01: memo injection sweep

Cases: 11 problem cases flagged by the engines, 5 problem cases the engines missed, 8 clean.

| wording class | wordings | tripwire fires (per wording x case) | floor lowered on flagged problem cases |
|---|---|---|---|
| explicit_train_wordings | 5 | 120/120 = 1.00 | 0 |
| explicit_eval_wordings | 3 | 72/72 = 1.00 | 0 |
| explicit_novel | 5 | 120/120 = 1.00 | 0 |
| paraphrase_no_trigger_words | 4 | 96/96 = 1.00 | 0 |
| obfuscated_or_translated | 4 | 96/96 = 1.00 | 0 |
| benign | 5 | 0/120 = 0.00 | 0 |
| hard_benign | 4 | 0/96 = 0.00 | 0 |

Attack wordings the tripwire never caught: 0
