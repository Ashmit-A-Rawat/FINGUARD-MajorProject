"""The LoRA adapter can be switched off so base and tuned arms share one model (no model download)."""

import contextlib
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from llm.inference.base import GenerationRequest, Message  # noqa: E402
from llm.inference.local_hf import LocalQwenProvider  # noqa: E402


class FakeInputs(dict[str, Any]):
    def to(self, _device: str) -> "FakeInputs":
        return self


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(
        self, messages: Any, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        return "prompt"

    def __call__(self, text: str, return_tensors: str) -> FakeInputs:
        return FakeInputs(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, tokens: Any, skip_special_tokens: bool) -> str:
        return "{}"


class FakeNet:
    def __init__(self) -> None:
        self.adapter_disabled_calls = 0

    @contextlib.contextmanager
    def disable_adapter(self):  # type: ignore[no-untyped-def]
        self.adapter_disabled_calls += 1
        yield

    def generate(self, **kwargs: Any) -> Any:
        return torch.tensor([[1, 2, 3, 4, 5]])


def make(adapter: str) -> tuple[LocalQwenProvider, FakeNet]:
    provider = LocalQwenProvider("dummy", "cpu", False, adapter)
    net = FakeNet()
    provider._net, provider._tokenizer, provider.device = net, FakeTokenizer(), "cpu"
    return provider, net


REQUEST = GenerationRequest(messages=[Message(role="user", content="hi")], max_new_tokens=4)


def test_adapter_on_by_default_does_not_disable() -> None:
    provider, net = make("some/adapter")
    provider.generate(REQUEST)
    assert net.adapter_disabled_calls == 0


def test_use_adapter_false_runs_base_model() -> None:
    provider, net = make("some/adapter")
    provider.use_adapter = False
    provider.generate(REQUEST)
    assert net.adapter_disabled_calls == 1


def test_without_an_adapter_the_switch_is_a_no_op() -> None:
    provider, net = make("")
    provider.use_adapter = False
    result = provider.generate(REQUEST)
    assert net.adapter_disabled_calls == 0
    assert result.completion_tokens == 2 and not result.is_mock
