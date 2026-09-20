import pytest
from kb_helpers import HashingEmbedder


@pytest.fixture(scope="session")
def embedder() -> HashingEmbedder:
    return HashingEmbedder()
