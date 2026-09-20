from collections.abc import Iterator
from pathlib import Path

import pytest
from api_helpers import SECRET, Api, FakeClock, HashingEmbedder
from fastapi.testclient import TestClient

from agents.context import AgentContext, build_context
from backend.app.core.config import Settings
from backend.app.core.security import Role
from backend.app.main import create_app
from backend.app.services.engine import EngineService
from data_pipeline.synthetic.config import GeneratorConfig
from data_pipeline.synthetic.generate import generate_dataset, write_dataset
from llm.inference.mock import MockLLMProvider


@pytest.fixture(scope="session")
def engine_ctx(tmp_path_factory: pytest.TempPathFactory) -> AgentContext:
    path = tmp_path_factory.mktemp("api_data")
    write_dataset(
        generate_dataset(GeneratorConfig(seed=5, n_customers=300, n_transactions=9000)), path
    )
    return build_context(path, HashingEmbedder(), MockLLMProvider(), clock=FakeClock())


@pytest.fixture()
def api(tmp_path: Path, engine_ctx: AgentContext) -> Iterator[Api]:
    settings = Settings(
        app_env="test",
        api_secret_key=SECRET,
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        engine_enabled=False,
        cors_origins="http://localhost:5173",
    )
    engine = EngineService.from_context(engine_ctx)
    app = create_app(settings, engine)
    with TestClient(app) as client:
        helper = Api(client, engine)
        helper.add_user("alice", Role.ANALYST)
        helper.add_user("bob", Role.ANALYST)
        helper.add_user("audrey", Role.AUDITOR)
        helper.add_user("root", Role.ADMIN)
        yield helper
    engine.shutdown()


@pytest.fixture()
def busy_customer(engine_ctx: AgentContext) -> str:
    store = engine_ctx.store
    return max(store.transactions_by_customer, key=lambda c: len(store.transactions_by_customer[c]))
