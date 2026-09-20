import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline.synthetic.anomalies import plan_anomaly_quotas
from data_pipeline.synthetic.config import (
    ANOMALY_TYPES,
    PRESETS,
    GeneratorConfig,
    config_for_preset,
)
from data_pipeline.synthetic.generate import SyntheticDataset, generate_dataset, write_dataset
from data_pipeline.synthetic.variations import apply_name_variation
from scripts.generate_synthetic_data import main as cli_main

SMALL = {"n_customers": 300, "n_transactions": 6000}


@pytest.fixture(scope="module")
def dataset() -> SyntheticDataset:
    return generate_dataset(GeneratorConfig(seed=7, **SMALL))


def test_presets_match_specification() -> None:
    assert PRESETS["small"] == (1_000, 20_000)
    assert PRESETS["medium"] == (10_000, 250_000)
    assert PRESETS["large"] == (100_000, 1_000_000)
    with pytest.raises(ValueError):
        config_for_preset("huge")


def test_mixes_must_sum_to_one() -> None:
    bad = {t: 0.5 for t in ANOMALY_TYPES}
    with pytest.raises(ValueError):
        GeneratorConfig(anomaly_mix=bad)


def test_generated_dataset_passes_validation(dataset: SyntheticDataset) -> None:
    assert dataset.validation.ok, dataset.validation.errors


def test_same_seed_is_bit_for_bit_reproducible(dataset: SyntheticDataset) -> None:
    again = generate_dataset(GeneratorConfig(seed=7, **SMALL))
    for name, df in dataset.tables.items():
        pd.testing.assert_frame_equal(df, again.tables[name], obj=name)


def test_different_seed_changes_data(dataset: SyntheticDataset) -> None:
    other = generate_dataset(GeneratorConfig(seed=8, **SMALL))
    assert not other.tables["transactions"]["amount"].equals(
        dataset.tables["transactions"]["amount"]
    )


def test_exact_row_counts(dataset: SyntheticDataset) -> None:
    assert len(dataset.tables["customers"]) == 300
    assert len(dataset.tables["transactions"]) == 6000


def test_anomaly_counts_match_plan(dataset: SyntheticDataset) -> None:
    cfg = GeneratorConfig(seed=7, **SMALL)
    labels = dataset.tables["transaction_labels"]
    counts = labels[labels["is_anomaly"]]["anomaly_type"].value_counts().to_dict()
    assert counts == plan_anomaly_quotas(cfg)
    assert labels["is_anomaly"].sum() == round(cfg.anomaly_rate * cfg.n_transactions)


def test_labels_are_not_leaked_into_transactions(dataset: SyntheticDataset) -> None:
    leaked = {"is_anomaly", "anomaly_type", "episode_id", "discrepancy_type"}
    assert not leaked & set(dataset.tables["transactions"].columns)
    assert not leaked & set(dataset.tables["ledger"].columns)


def test_unusual_time_anomalies_are_at_night(dataset: SyntheticDataset) -> None:
    tx = dataset.tables["transactions"].merge(
        dataset.tables["transaction_labels"], on="transaction_id"
    )
    hours = pd.to_datetime(tx.loc[tx["anomaly_type"] == "unusual_time", "timestamp"]).dt.hour
    assert hours.between(1, 4).all()


def test_geo_change_goes_to_unusual_country(dataset: SyntheticDataset) -> None:
    from data_pipeline.synthetic.reference_data import UNUSUAL_COUNTRIES

    tx = dataset.tables["transactions"].merge(
        dataset.tables["transaction_labels"], on="transaction_id"
    )
    assert (
        tx.loc[tx["anomaly_type"] == "geo_change", "receiver_country"].isin(UNUSUAL_COUNTRIES).all()
    )


def test_repeated_transfers_share_receiver_and_amount(dataset: SyntheticDataset) -> None:
    tx = dataset.tables["transactions"].merge(
        dataset.tables["transaction_labels"], on="transaction_id"
    )
    rep = tx[tx["anomaly_type"] == "repeated_transfers"]
    for _, group in rep.groupby("episode_id"):
        assert group["receiver"].nunique() == 1
        assert group["amount"].nunique() == 1


def test_ledger_discrepancies_are_real_and_labelled(dataset: SyntheticDataset) -> None:
    tx = dataset.tables["transactions"].set_index("transaction_id")
    ledger = dataset.tables["ledger"]
    labels = dataset.tables["ledger_labels"]
    amount = labels[labels["discrepancy_type"] == "contradictory_amount"].merge(
        ledger, on="ledger_id"
    )
    assert (
        amount["posted_amount"].to_numpy()
        != tx.loc[amount["transaction_id_x"], "amount"].to_numpy()
    ).all()
    missing = labels[labels["discrepancy_type"] == "missing_ledger_entry"]["transaction_id"]
    assert not set(missing) & set(ledger["transaction_id"])
    dup = labels[labels["discrepancy_type"] == "duplicate_posting"]["transaction_id"]
    assert (ledger["transaction_id"].value_counts().reindex(dup) == 2).all()
    no_ref = labels[labels["discrepancy_type"] == "missing_reference"]["ledger_id"]
    assert ledger.set_index("ledger_id").loc[no_ref, "reference_id"].isna().all()


def test_clean_ledger_rows_match_transactions(dataset: SyntheticDataset) -> None:
    labels = dataset.tables["ledger_labels"]
    tx = dataset.tables["transactions"]
    clean = dataset.tables["ledger"].merge(tx, on="transaction_id", suffixes=("_l", "_t"))
    clean = clean[~clean["transaction_id"].isin(labels["transaction_id"])]
    assert len(clean) > 5000
    assert (clean["posted_amount"] == clean["amount"]).all()
    assert (clean["currency_l"] == clean["currency_t"]).all()
    assert (clean["reference_id_l"] == clean["reference_id_t"]).all()


def test_kyc_variation_labels_describe_actual_differences(dataset: SyntheticDataset) -> None:
    kyc = dataset.tables["kyc_records"].merge(
        dataset.tables["kyc_labels"], on=["document_id", "customer_id"]
    )
    cust = dataset.tables["customers"].set_index("customer_id")
    for _, row in kyc.iterrows():
        c = cust.loc[row["customer_id"]]
        name_same = row["name"] == c["name"]
        if row["variation_type"] == "none":
            assert (
                name_same
                and row["address"] == c["address"]
                and row["date_of_birth"] == c["date_of_birth"]
            )
        elif row["variation_type"] == "address_mismatch":
            assert row["address"] != c["address"]
        elif row["variation_type"] == "dob_conflict":
            assert row["date_of_birth"] != c["date_of_birth"]
        else:
            assert not name_same, row["variation_type"]


def test_duplicate_and_namesake_entities(dataset: SyntheticDataset) -> None:
    ent = dataset.tables["entity_labels"]
    cust = dataset.tables["customers"].set_index("customer_id")
    dups = ent[ent["relation"] == "duplicate_of"]
    assert len(dups) == 6
    for _, d in dups.iterrows():
        orig = ent.set_index("customer_id").loc[d["related_customer_id"]]
        assert orig["entity_id"] == d["entity_id"]
        assert (
            cust.loc[d["customer_id"], "date_of_birth"]
            == cust.loc[d["related_customer_id"], "date_of_birth"]
        )
    same = ent[ent["relation"] == "namesake_of"]
    assert len(same) == 3
    for _, s in same.iterrows():
        assert (
            s["entity_id"]
            != ent.set_index("customer_id").loc[s["related_customer_id"], "entity_id"]
        )


@pytest.mark.parametrize(
    "kind", ["typo", "abbreviation", "transliteration", "middle_name_diff", "name_order_swap"]
)
def test_name_variation_always_changes_the_name(kind: str) -> None:
    rng = np.random.default_rng(0)
    for first, middle, last in [
        ("Ava", None, "Adams"),
        ("Mohammed", "Grace", "Smith"),
        ("Li", None, "Zhang"),
    ]:
        base = " ".join(p for p in (first, middle, last) if p)
        for _ in range(50):
            name, applied = apply_name_variation(kind, first, middle, last, rng)
            assert name != base, (kind, applied)


def test_write_dataset_and_manifest(dataset: SyntheticDataset, tmp_path: Path) -> None:
    write_dataset(dataset, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["validation"]["ok"] is True
    assert "SYNTHETIC" in manifest["notice"]
    assert set(manifest["file_sha256"]) == {f"{n}.csv" for n in dataset.tables}
    assert (tmp_path / "SYNTHETIC_DATA_NOTICE.txt").exists()


def test_cli_refuses_large_preset_without_flag(tmp_path: Path) -> None:
    assert cli_main(["--preset", "large", "--output-dir", str(tmp_path)]) == 2
    assert not list(tmp_path.iterdir())
