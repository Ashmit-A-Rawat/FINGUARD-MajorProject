"""Heuristic detection of instruction-like text inside documents.

FLAGS, DOES NOT DELETE. This is a tripwire for reviewers and tests, not a defence: attackers can
rephrase. The real defence is architectural: retrieved text is always passed to a model as fenced
DATA, never as instructions (Phase 9/10 prompt design and the guardrails).
"""

import re

PATTERNS: dict[str, re.Pattern[str]] = {
    "override_previous_instructions": re.compile(
        r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|any)\b"
        r"[^.\n]{0,30}\b(instructions?|rules?|prompts?|guidelines?|policy|policies)\b",
        re.I,
    ),
    "role_reassignment": re.compile(
        r"\byou are (now|no longer)\b|\bact as (the|a|an)\b|\bpretend to be\b", re.I
    ),
    "fake_role_tags": re.compile(
        r"</?\s*(system|assistant|user)\s*>|<\|im_(start|end)\|>|\[/?INST\]", re.I
    ),
    "prompt_disclosure": re.compile(
        r"\b(reveal|print|show|repeat)\b[^.\n]{0,30}\b(system prompt|instructions)\b", re.I
    ),
    "forced_decision": re.compile(
        r"\b(always|must|should)\b[^.\n]{0,40}\b(mark|set|classify|record|answer|report)\b[^.\n]{0,40}"
        r"\b(clear|approved|no discrepanc\w+|legitimate)\b",
        re.I,
    ),
    "suppress_reporting": re.compile(
        r"\bdo not\b[^.\n]{0,30}\b(flag|escalate|report|review|mention)\b", re.I
    ),
}


def injection_flags(text: str) -> list[str]:
    return [name for name, pattern in PATTERNS.items() if pattern.search(text)]
