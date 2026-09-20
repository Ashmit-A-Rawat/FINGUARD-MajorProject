"""LLM provider abstraction. The rest of FIN-GUARD depends only on this interface."""

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerationRequest(BaseModel):
    messages: list[Message]
    max_new_tokens: int = Field(768, ge=1)
    temperature: float = Field(0.0, ge=0.0)  # 0 = greedy / deterministic
    seed: int | None = 0


class GenerationResult(BaseModel):
    text: str
    provider: str
    model: str
    is_mock: bool  # True means NO language model produced this text
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ModelNotAvailableError(RuntimeError):
    """Raised when a local model is not present and downloading has not been allowed."""


class LLMProvider(ABC):
    name: str
    model: str
    is_mock: bool = False

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return the model's raw text. Validation happens above this layer."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release model memory."""
