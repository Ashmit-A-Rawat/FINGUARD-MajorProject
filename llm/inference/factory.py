"""Build the configured provider. The only place that knows about concrete provider classes."""

from backend.app.core.config import Settings, get_settings
from llm.inference.base import LLMProvider
from llm.inference.local_hf import LocalMistralProvider, LocalQwenProvider
from llm.inference.mock import MockLLMProvider


def create_provider(settings: Settings | None = None) -> LLMProvider:
    s = settings or get_settings()
    name = s.llm_provider.lower()
    if name == "mock":
        return MockLLMProvider()
    if name == "qwen":
        return LocalQwenProvider(s.llm_model, s.llm_device, s.llm_allow_download)
    if name == "mistral":
        return LocalMistralProvider(s.llm_model, s.llm_device, s.llm_allow_download)
    raise ValueError(f"unknown LLM_PROVIDER {s.llm_provider!r}; use mock, qwen or mistral")
