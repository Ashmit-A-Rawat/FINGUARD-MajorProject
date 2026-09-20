# Data Architecture

> **All data in this project is SYNTHETIC.** It contains no real people, accounts or transactions.
> Results on it do not represent real-bank performance.

Code: `backend/app/schemas/domain.py` (schemas), `data_pipeline/synthetic/` (generator),
`scripts/generate_synthetic_data.py` (CLI). Output goes to `data/synthetic/<preset>/` (git-ignored).

## Design decisions

- **Labels live in separate files.** Ground truth (`*_labels.csv`) is never a column of the
  feature tables, so it cannot leak into model inputs. Tests enforce this.
- **Deterministic.** One `SeedSequence(seed)` spawns an independent RNG per stage (customers,
  profiles, transactions, anomalies, ledger). Same seed + config gives identical tables
  (test: bit-for-bit). The observation window is fixed (`window_end = 2025-06-30`), never "now".
- **Every record has `is_synthetic=True`**, and the schema refuses `False`. Each output folder has a
  `SYNTHETIC_DATA_NOTICE.txt` and a `manifest.json` (config, counts, realised label counts,
  validation result, per-file SHA-256).
- **Realised counts are reported**, not assumed. If an anomaly quota could not be met the manifest shows it.

## Presets

| Preset | Customers | Transactions | Notes |
|---|---|---|---|
| small | 1,000 | 20,000 | default; generates in about 2 s |
| medium | 10,000 | 250,000 | |
| large | 100,000 | 1,000,000 | requires `--allow-large` |

## Data dictionary

**customers.csv**: `customer_id`, `name`, `alternate_names` ('|'-joined), `date_of_birth`, `address`,
`country`, `occupation`, `account_type` (personal/business/savings), `account_open_date`,
`account_age_days` (as of window end), `kyc_status` (verified/pending/expired/rejected), `is_synthetic`.
`kyc_status` is `expired` if the document expired before the window end, otherwise drawn
independently of any KYC-variation label.

**kyc_records.csv**: `document_id`, `customer_id`, `name`, `date_of_birth`, `address`, `document_type`,
`document_number`, `issue_date`, `expiry_date`, `is_synthetic`. One document per customer, which may
carry one controlled variation.

**transactions.csv**: `transaction_id`, `customer_id`, `timestamp`, `amount` (local currency),
`currency`, `transaction_type` (transfer_out/transfer_in/payment/withdrawal/deposit), `sender`,
`receiver`, `sender_country`, `receiver_country`, `channel` (mobile/web/branch/atm/pos),
`reference_id`, `is_synthetic`. Sorted by timestamp.

**ledger.csv**: `ledger_id`, `transaction_id`, `posted_amount`, `currency`, `posting_timestamp`,
`settlement_status` (settled/pending/failed), `reference_id`, `is_synthetic`. Normally one row per
transaction.

**Labels (ground truth, evaluation only)**

| File | Columns |
|---|---|
| transaction_labels.csv | `transaction_id`, `is_anomaly`, `anomaly_type` (or `none`), `episode_id` |
| ledger_labels.csv | `transaction_id`, `ledger_id` (blank if the entry is missing), `discrepancy_type` |
| kyc_labels.csv | `document_id`, `customer_id`, `variation_type` |
| entity_labels.csv | `customer_id`, `entity_id`, `relation` (unique/duplicate_of/namesake_of), `related_customer_id` |

## Normal behaviour model

Per customer: a typical amount (occupation base x account-type multiplier x lognormal noise, scaled
to home currency by *illustrative constant* FX rates), lognormal amount spread, a preferred hour of
day (normal, sd 2.5 h), a set of 3-12 regular counterparties (85% of transactions) plus occasional
new ones, ~10% foreign counterparties, and channel by type. Customer activity volume is lognormal
(heavy-tailed). About 10% of accounts open inside the window and start transacting on opening.
Transaction type mix: 35% transfer_out, 30% payment, 15% transfer_in, 10% withdrawal, 10% deposit.

## Anomaly injection (default rate 2% of transactions)

| Type | Mechanism | Share |
|---|---|---|
| high_value | amount x U(8,30) on an outgoing transfer | 20% |
| burst | episode of 5-12 transfers within 10 min to one new counterparty | 15% |
| unusual_time | moved to 01:00-04:59 same day | 15% |
| geo_change | counterparty in a country from `UNUSUAL_COUNTRIES`, via web | 12% |
| new_counterparty | never-seen counterparty, amount x U(2,4) | 12% |
| abnormal_velocity | episode of 15-30 transactions in 24 h | 10% |
| repeated_transfers | 4-8 identical amounts to one new receiver, 1-3 days apart | 16% |

Each row carries exactly one label; episode rows share an `episode_id`. Anomaly rows are only ever
*behavioural*; data-quality problems are in the ledger.

## Ledger discrepancies (default 1.5% of transactions)

contradictory_amount (30%), missing_reference (20%, ledger reference blanked), currency_mismatch
(10%), missing_ledger_entry (15%), duplicate_posting (15%), late_posting (8-20 days, 10%).

## KYC variations and entity structure

Per document (default mix): none 79%, typo 4%, abbreviation 3%, transliteration 3%,
middle_name_diff 3%, name_order_swap 2%, address_mismatch 5%, dob_conflict 1%. If a variation is
impossible for a name (e.g. no transliteration exists) the generator falls back to a typo and
**labels what was actually applied**; tests verify every label against the real difference.
2% of customers are **duplicates** (same person, same DOB, name spelling varied, address same or
altered, new customer id) and 1% are **namesakes** (same first and last name, different person).
Ids are assigned after shuffling so they never reveal structure.

## Validation (`data_pipeline/synthetic/validation.py`)

Pydantic validation of every row; unique ids; exact row counts; referential integrity between
tables; timestamps in window and sorted; label ids and label vocabularies. The result is recorded
in `manifest.json`, and the CLI exits non-zero if validation fails.

## Assumptions and limitations

- Distributions are simple and invented; real banking behaviour is far more varied (seasonality,
  payroll cycles, merchant categories, network structure between customers).
- Anomalies are injected with clear signatures, so detectors may score better here than on real data.
  Compare methods *against each other*, not against real-world expectations.
- Name/address pools are small (about 50 first names, 50 surnames), so accidental name collisions
  are common; they act as natural hard negatives for KYC matching.
- Names skew Western/South-Asian and transliteration variants are hand-picked, not a full model of
  transliteration.
- FX rates are illustrative constants. Amounts are floats, not decimals.
- Ledger has no independent settlement process; discrepancies are injected, not emergent.
- Customers are independent, so no fraud rings, mule accounts or shared counterparties are modelled.
- Ledger labels do not cover a discrepancy stacked on top of another (one label per transaction).

# Phase 3: Data pipeline

Code: `data_pipeline/` (`pipeline.py` orchestrates). Run with `make pipeline` or
`python scripts/run_pipeline.py --preset small`. Output: `data/processed/<preset>/`
(`pipeline_report.json`, `quarantine.json`, `customers_normalized.csv`, `kyc_normalized.csv`).

```
CSV files -> integrity check -> raw read (all strings) -> cleaning -> schema validation
          -> referential checks -> normalization -> consolidated store -> canonical case
```

## Principles ("no silent data corruption")

1. **Integrity first.** If `manifest.json` exists, every file's SHA-256 is checked; a mismatch raises `IntegrityError`.
2. **Nothing disappears.** A row that fails validation is *quarantined* with `table`, `row_number`,
   `reason`, `detail` and the full record. Reasons: `missing_primary_key`, `duplicate_key_conflict`,
   `schema_validation_failed`, `orphan_reference`, `normalization_failed`. Byte-identical duplicate rows
   are dropped and counted (`exact_duplicates_dropped`).
3. **Every cleaning change is counted** per table (`trim_whitespace`, `collapse_whitespace`,
   `unicode_nfc`, `null_token_to_none`). Cleaning never changes meaning: no case changes, no
   accent stripping. Null tokens are whole-field only (`""`, `nan`, `null`, `none`, `n/a`); `NA`
   is kept because it is a valid country code.
4. **Normalization adds, never replaces.** Each canonical object embeds the raw record plus
   normalized forms (`NormalizedName`, `NormalizedAddress`, normalized document number). The raw value
   stays available to the KYC engine and to human reviewers.
5. **Unparseable is flagged, not guessed.** Addresses that do not match the expected structure are
   kept with `parsed=False` and counted in the report. A name with no usable tokens is quarantined.
6. **The report is data.** `PipelineReport` records rows read/accepted, cleaning actions,
   quarantine counts, unparsed addresses, and ledger facts (transactions with no / several ledger rows).

## Normalization rules

- **Names:** accent strip, casefold, remove `.` and punctuation (keeps `-` and `'`), collapse spaces.
  `Last, First Middle` (comma only) becomes `first middle last` with `reordered=True`. Flags `has_initials`.
  Token order is otherwise kept, so `Last First` without a comma is *not* guessed.
- **Addresses:** `number street, city, CC` is parsed; street suffix standardised
  (`Street/St.` to `st`, `Avenue/Ave` to `ave`, etc.); country upper-cased.
- **Document numbers:** upper-case alphanumerics only.
- **Dates and amounts:** parsed by the Pydantic schemas (ISO dates only; ambiguous `DD/MM` vs `MM/DD`
  formats are rejected rather than guessed).

## Canonical case (`CanonicalCase`)

The single input object for the KYC, anomaly and reconciliation engines: customer (raw + normalized),
KYC documents, focus transactions with all their ledger records, and context transactions.
- Focus defaults to the customer's latest transaction. `as_of` = latest focus timestamp.
- Context = the customer's other transactions in `[as_of - context_days, as_of]`: **no future data**.
- **No ground-truth labels** appear anywhere in a case (tested by walking the serialised JSON).
- Ledger records are attached as observed facts (0, 1 or many); judging discrepancies is Phase 6.

## Scope notes

- Entity resolution (duplicate / namesake detection) is *not* done here; it is Phase 4. Consolidation
  here means linking records and building cases.
- The synthetic data is already clean, so the pipeline's cleaning and quarantine paths are exercised
  by unit tests that inject dirt (orphans, negative amounts, bad enums, conflicting keys, messy whitespace).
- All tables are held in memory. Fine for small/medium presets; `large` would need a streaming or
  database-backed store (Phase 3+ database work is deferred until needed).

## KYC benchmark data

The KYC benchmark (`evaluation/benchmarks/kyc_benchmark.py`) reuses `generate_customers` with a
balanced variation mix and higher duplicate/namesake rates (6% each). It is separate from the main
dataset presets and needs no transactions. See [ml-architecture](ml-architecture.md#kyc-entity-resolution-kyc)
and the [evaluation plan](../research/evaluation-plan.md).
