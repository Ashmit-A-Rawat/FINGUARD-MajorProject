"""Generate structured output with validation and a bounded repair loop."""

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel

from guardrails.schema_validator import parse_structured
from llm.inference.base import GenerationRequest, GenerationResult, LLMProvider, Message

M = TypeVar("M", bound=BaseModel)


@dataclass
class Attempt:
    text: str
    error: str | None  # None = this attempt validated
    result: GenerationResult


@dataclass
class StructuredResult(Generic[M]):
    value: M | None
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None

    @property
    def is_mock(self) -> bool:
        return any(a.result.is_mock for a in self.attempts)

    @property
    def total_latency_s(self) -> float:
        return sum(a.result.latency_s for a in self.attempts)


def _repair_message(error: str) -> Message:
    return Message(
        role="user",
        content=(
            f"Your previous reply was rejected: {error}\n"
            "Reply again with ONLY the corrected JSON object, no other text."
        ),
    )


def generate_structured(
    provider: LLMProvider,
    messages: list[Message],
    model: type[M],
    *,
    max_attempts: int = 3,
    max_new_tokens: int = 768,
) -> StructuredResult[M]:
    """Ask, validate, and on failure show the model the error and ask again (bounded).

    Every attempt is recorded so callers can log it. Failure returns ``value=None``; the caller
    must route to human review, never guess a value.
    """
    conversation = list(messages)
    result: StructuredResult[M] = StructuredResult(value=None)
    for _ in range(max_attempts):
        generated = provider.generate(
            GenerationRequest(messages=conversation, max_new_tokens=max_new_tokens)
        )
        parsed = parse_structured(generated.text, model)
        result.attempts.append(Attempt(generated.text, parsed.error, generated))
        if parsed.ok:
            result.value = parsed.value
            return result
        conversation = [
            *conversation,
            Message(role="assistant", content=generated.text),
            _repair_message(parsed.error or "unknown error"),
        ]
    return result
