import numpy as np
import pandas as pd

from anomaly_detection.features.builder import MIN_HISTORY, build_features


def test_features_are_finite_and_aligned(tx: pd.DataFrame, customers: pd.DataFrame) -> None:
    ff = build_features(tx, customers)
    assert len(ff.frame) == len(tx)
    assert np.isfinite(ff.frame.to_numpy()).all()
    assert set(ff.basic_columns) < set(ff.engineered_columns)


def test_features_are_causal_truncation_invariance(
    tx: pd.DataFrame, customers: pd.DataFrame
) -> None:
    """Recomputing on data cut at time T must reproduce every earlier row exactly.

    If any feature peeked at later rows, dropping the future would change these values.
    """
    full = build_features(tx, customers)
    cut = tx["timestamp"].quantile(0.5)
    head = tx[tx["timestamp"] <= cut].reset_index(drop=True)
    truncated = build_features(head, customers)
    n = len(head)
    pd.testing.assert_frame_equal(
        full.frame.iloc[:n].reset_index(drop=True), truncated.frame, check_exact=False, atol=1e-5
    )
    np.testing.assert_array_equal(
        full.categorical["cp_bucket"][:n], truncated.categorical["cp_bucket"]
    )


def test_first_transaction_of_a_customer_has_no_history(
    tx: pd.DataFrame, customers: pd.DataFrame
) -> None:
    ff = build_features(tx, customers)
    first = ~tx["customer_id"].duplicated().to_numpy()
    f = ff.frame[first]
    assert (f["log_n_prior_tx"] == 0).all()
    assert (f["z_amount_vs_history"] == 0).all() and (f["is_new_counterparty"] == 1).all()
    assert (f["log_count_24h"] == 0).all() and (f["log_prior_counterparty_count"] == 0).all()


def test_counterparty_novelty_matches_manual_computation(
    tx: pd.DataFrame, customers: pd.DataFrame
) -> None:
    ff = build_features(tx, customers)
    outgoing = tx["transaction_type"].isin(["transfer_out", "payment", "withdrawal"])
    cp = np.where(outgoing, tx["receiver"], tx["sender"])
    expected = ~pd.DataFrame({"c": tx["customer_id"], "p": cp}).duplicated().to_numpy()
    np.testing.assert_array_equal(ff.frame["is_new_counterparty"].to_numpy() == 1, expected)


def test_window_count_matches_brute_force(tx: pd.DataFrame, customers: pd.DataFrame) -> None:
    ff = build_features(tx, customers)
    busiest = tx["customer_id"].value_counts().index[0]
    rows = tx.index[tx["customer_id"] == busiest]
    times = tx.loc[rows, "timestamp"].astype("int64").to_numpy() // 10**9
    for k in (5, 20, len(rows) - 1):
        prior = times[:k]
        expected = ((times[k] - prior) <= 3600).sum()
        assert np.isclose(np.expm1(ff.frame.loc[rows[k], "log_count_1h"]), expected)


def test_history_statistics_ignore_the_current_row(
    tx: pd.DataFrame, customers: pd.DataFrame
) -> None:
    ff = build_features(tx, customers)
    busiest = tx["customer_id"].value_counts().index[0]
    rows = tx.index[tx["customer_id"] == busiest]
    fx = tx.loc[rows, "amount"].to_numpy()
    k = 30
    from data_pipeline.synthetic.reference_data import UNITS_PER_USD

    usd = np.log1p(fx / tx.loc[rows, "currency"].map(UNITS_PER_USD).to_numpy())
    prior_mean, prior_std = usd[:k].mean(), max(usd[:k].std(), 0.25)
    expected = (usd[k] - prior_mean) / prior_std
    assert k >= MIN_HISTORY
    assert np.isclose(ff.frame.loc[rows[k], "z_amount_vs_history"], expected, atol=1e-3)


def test_sequences_end_with_self_and_never_look_forward(
    tx: pd.DataFrame, customers: pd.DataFrame
) -> None:
    ff = build_features(tx, customers, seq_len=8)
    seq = ff.seq_index
    assert (seq[:, -1] == np.arange(len(tx))).all()
    ts = tx["timestamp"].astype("int64").to_numpy()
    cust = ff.customer_codes
    for i in np.random.default_rng(0).integers(0, len(tx), 300):
        members = seq[i][seq[i] >= 0]
        assert (cust[members] == cust[i]).all()
        assert (ts[members] <= ts[i]).all()
        assert (np.diff(ts[members]) >= 0).all()
    assert (seq[~tx["customer_id"].duplicated().to_numpy()][:, :-1] == -1).all()


def test_unsorted_input_is_rejected(tx: pd.DataFrame, customers: pd.DataFrame) -> None:
    import pytest

    with pytest.raises(ValueError):
        build_features(tx.iloc[::-1].reset_index(drop=True), customers)
