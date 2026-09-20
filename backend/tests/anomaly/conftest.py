import pandas as pd
import pytest

from data_pipeline.synthetic.config import GeneratorConfig
from data_pipeline.synthetic.generate import SyntheticDataset, generate_dataset


@pytest.fixture(scope="session")
def dataset() -> SyntheticDataset:
    return generate_dataset(GeneratorConfig(seed=21, n_customers=250, n_transactions=8000))


@pytest.fixture(scope="session")
def tx(dataset: SyntheticDataset) -> pd.DataFrame:
    frame = dataset.tables["transactions"].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)


@pytest.fixture(scope="session")
def customers(dataset: SyntheticDataset) -> pd.DataFrame:
    return dataset.tables["customers"]
