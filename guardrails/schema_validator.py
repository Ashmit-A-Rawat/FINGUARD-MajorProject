"""Extract JSON from raw model text and validate it against a Pydantic model.

Models often wrap JSON in prose or code fences, or truncate it. This module never guesses:
it either returns a validated object or a precise error that can be shown to the model for repair.
"""

import json
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)


@dataclass
class ParseResult(Generic[M]):
    value: M | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.value is not None


def extract_json_object(text: str) -> str:
    """Return the first balanced top-level {...} in ``text`` (string- and escape-aware).

    Raises ValueError if there is none or it is unterminated (truncated output).
    """
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in the output")
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("JSON object is not terminated (output was probably truncated)")


def parse_structured(text: str, model: type[M]) -> ParseResult[M]:
    try:
        payload = json.loads(extract_json_object(text))
    except (ValueError, json.JSONDecodeError) as exc:
        return ParseResult(None, f"invalid JSON: {exc}")
    try:
        return ParseResult(model.model_validate(payload), None)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'root'}: {e['msg']}" for e in exc.errors()[:8]
        )
        return ParseResult(None, f"schema validation failed: {details}")
