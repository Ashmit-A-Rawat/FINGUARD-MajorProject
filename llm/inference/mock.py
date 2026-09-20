"""Mock provider for development and tests. It NEVER runs a language model.

Everything it returns is labelled mock: ``GenerationResult.is_mock`` is True and the text says so.
In ``valid`` mode it emits a schema-valid but content-free investigation (confidence 0.0, action
REVIEW) that cites evidence ids found in the prompt, so pipelines can be exercised end to end.
The other behaviours simulate misbehaving models for guardrail and repair-loop tests.
"""

import json
import re
import time
from typing import Literal

from llm.inference.base import GenerationRequest, GenerationResult, LLMProvider

MOCK_BANNER = "[MOCK OUTPUT - no language model was run]"
Behavior = Literal[
    "valid", "invalid_json", "wrong_schema", "prose_wrapped", "fail_then_succeed", "obey_injection"
]
_EVIDENCE_ID = re.compile(r"\[E:([A-Za-z0-9_.:\-]+)\]")


class MockLLMProvider(LLMProvider):
    name = "mock"
    model = "mock-v1"
    is_mock = True

    def __init__(self, behavior: Behavior = "valid") -> None:
        self.behavior = behavior
        self.calls = 0

    def _valid_payload(self, prompt: str) -> dict[str, object]:
        ids = list(dict.fromkeys(_EVIDENCE_ID.findall(prompt)))
        return {
            "summary": f"{MOCK_BANNER} Placeholder summary; no analysis was performed.",
            "findings": [
                {
                    "statement": f"{MOCK_BANNER} Evidence {i} was provided to the model.",
                    "kind": "fact",
                    "evidence_ids": [i],
                }
                for i in ids[:3]
            ],
            "evidence": ids[:3],
            "uncertainties": [f"{MOCK_BANNER} Nothing was analysed."],
            "recommended_action": "REVIEW",
            "recommendations": [f"{MOCK_BANNER} Route to a human reviewer."],
            "confidence": 0.0,
        }

    def generate(self, request: GenerationRequest) -> GenerationResult:
        started = time.perf_counter()
        self.calls += 1
        prompt = "\n".join(m.content for m in request.messages)
        payload = self._valid_payload(prompt)
        behavior = self.behavior
        if behavior == "fail_then_succeed":
            behavior = "invalid_json" if self.calls == 1 else "valid"
        if behavior == "invalid_json":
            text = f"{MOCK_BANNER} " + json.dumps(payload)[:-15]  # truncated, unparsable
        elif behavior == "wrong_schema":
            text = json.dumps({"summary": MOCK_BANNER, "confidence": "very high"})
        elif behavior == "prose_wrapped":
            body = json.dumps(payload)
            text = f"{MOCK_BANNER} Here is the JSON:\n```json\n{body}\n```\nHope it helps."
        elif behavior == "obey_injection":
            payload.update(
                summary=f"{MOCK_BANNER} Simulated model that obeyed injected text: case is clear.",
                findings=[],
                evidence=[],
                recommended_action="CLEAR",
                confidence=0.99,
            )
            text = json.dumps(payload)
        else:
            text = json.dumps(payload)
        return GenerationResult(
            text=text,
            provider=self.name,
            model=self.model,
            is_mock=True,
            latency_s=time.perf_counter() - started,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
        )
