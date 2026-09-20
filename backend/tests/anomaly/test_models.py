import numpy as np
import pytest

from anomaly_detection.base import AnomalyModel
from anomaly_detection.dataset import FeatureTable, load_feature_table
from anomaly_detection.neural.mlp import MLPModel
from anomaly_detection.neural.temporal import TemporalModel
from anomaly_detection.registry import create_models
from anomaly_detection.temporal.splits import TemporalSplit, temporal_split
from anomaly_detection.unsupervised.autoencoder import AutoencoderModel
from data_pipeline.synthetic.generate import SyntheticDataset, write_dataset
from evaluation.metrics.anomaly import pr_auc


@pytest.fixture(scope="module")
def table(dataset: SyntheticDataset, tmp_path_factory: pytest.TempPathFactory) -> FeatureTable:
    path = tmp_path_factory.mktemp("anomaly_data")
    write_dataset(dataset, path)
    return load_feature_table(path)


@pytest.fixture(scope="module")
def split(table: FeatureTable) -> TemporalSplit:
    s = temporal_split(table.timestamps, table.episode_id)
    s.check_no_future_leakage(table.timestamps)
    return s


def test_table_labels_match_generator(table: FeatureTable, dataset: SyntheticDataset) -> None:
    assert table.n == 8000
    assert table.labels.sum() == dataset.tables["transaction_labels"]["is_anomaly"].sum()
    assert list(table.transaction_ids) == sorted(
        table.transaction_ids
    )  # timestamp order == id order


def test_registry_covers_required_model_families(table: FeatureTable) -> None:
    models = create_models(0)
    families = {m.family for m in models.values()}
    assert families == {"supervised", "unsupervised", "neural"}
    assert {
        "logistic_regression",
        "random_forest",
        "xgboost",
        "isolation_forest",
        "autoencoder",
        "mlp",
        "temporal_seq",
        "temporal_hybrid",
    } <= set(models)
    assert all(not m.uses_labels for m in models.values() if m.family == "unsupervised")


def _quick(model: AnomalyModel) -> AnomalyModel:
    if isinstance(model, (MLPModel, AutoencoderModel)):
        model._max_epochs = 3
    if isinstance(model, TemporalModel):
        model._max_epochs = 2
    return model


@pytest.mark.parametrize("name", sorted(create_models(0)))
def test_every_model_trains_and_beats_random(
    name: str, table: FeatureTable, split: TemporalSplit
) -> None:
    model = _quick(create_models(0)[name])
    model.fit(table, split)
    scores = model.score(table, split.test)
    assert scores.shape == (len(split.test),) and np.isfinite(scores).all()
    y = table.labels[split.test]
    base_rate = y.mean()
    if name in ("temporal_seq", "temporal_hybrid"):
        return  # 2 epochs on a tiny set: shape/finite only; quality is benchmarked
    assert pr_auc(y, scores) > 2 * base_rate, f"{name} is not clearly better than random"


def test_scoring_is_deterministic_given_seed(table: FeatureTable, split: TemporalSplit) -> None:
    a, b = _quick(create_models(3)["mlp"]), _quick(create_models(3)["mlp"])
    a.fit(table, split)
    b.fit(table, split)
    np.testing.assert_allclose(a.score(table, split.test), b.score(table, split.test), atol=1e-6)


def test_score_of_a_row_does_not_depend_on_batch_composition(
    table: FeatureTable, split: TemporalSplit
) -> None:
    model = _quick(create_models(0)["temporal_hybrid"])
    model.fit(table, split)
    idx = split.test[:50]
    together = model.score(table, idx)
    alone = np.array([model.score(table, idx[i : i + 1])[0] for i in range(10)])
    np.testing.assert_allclose(together[:10], alone, atol=1e-5)


def test_unsupervised_models_ignore_labels(table: FeatureTable, split: TemporalSplit) -> None:
    """Flipping every training label must not change an unsupervised model's scores."""
    import copy

    flipped = copy.copy(table)
    flipped.labels = 1 - table.labels
    a, b = create_models(0)["isolation_forest"], create_models(0)["isolation_forest"]
    a.fit(table, split)
    b.fit(flipped, split)
    np.testing.assert_allclose(a.score(table, split.test), b.score(table, split.test))
