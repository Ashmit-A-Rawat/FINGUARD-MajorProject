from pathlib import Path

import pytest
from agent_helpers import FakeClock, HashingEmbedder

from agents.context import AgentContext, build_context
from data_pipeline.synthetic.config import GeneratorConfig
from data_pipeline.synthetic.generate import generate_dataset, write_dataset
from llm.inference.mock import MockLLMProvider


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("agents_data")
    write_dataset(
        generate_dataset(GeneratorConfig(seed=5, n_customers=300, n_transactions=9000)), path
    )
    return path


@pytest.fixture(scope="session")
def ctx(data_dir: Path) -> AgentContext:
    return build_context(data_dir, HashingEmbedder(), MockLLMProvider(), clock=FakeClock())


@pytest.fixture(scope="session")
def busy_customer(ctx: AgentContext) -> str:
    store = ctx.store
    return max(store.transactions_by_customer, key=lambda c: len(store.transactions_by_customer[c]))
