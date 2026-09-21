"""Heuristic detection of instruction-like text inside documents.

FLAGS, DOES NOT DELETE. This is a tripwire for reviewers and tests, not a defence: attackers can
rephrase. The real defence is architectural: retrieved text is always passed to a model as fenced
DATA, never as instructions (Phase 9/10 prompt design and the guardrails).

Text is first NORMALISED (Unicode compatibility forms such as full-width letters, zero-width characters,
letter-spacing like 'i-g-n-o-r-e', digit-for-letter substitutions like 'd0 n0t'); patterns run on both the
raw and the normalised text. See docs/research/experiments.md (EXP-ADV-02) for measured recall and
false-positive rates and for what the patterns still miss.
"""

import re
import unicodedata

_OVERRIDE_VERBS = r"(?:(?<!to )|(?<=you to ))(?:ignore|ignoring|disregard|(?<!not )(?<!n't )forget|override|bypass|skip|neglect)"
_OVERRIDE_OBJECTS = (
    r"(?:instructions?|rules?|prompts?|guidelines?|guidance|polic(?:y|ies)|checks?|reviews?|evidence|"
    r"analysis|everything|above|before|previous|prior|earlier)"
)
_AI_TERMS = (
    r"(?:ai|assistant|model|bot|llm|approv\w*|officer|admin\w*|reviewer|analyst|system|compliance)"
)
_DECISIONS = (
    r"(?:clear|cleared|approved?|low[- ]risk|verified|normal|legitimate|no (?:issues?|discrepanc\w+|findings?)|"
    r"false positives?)"
)

PATTERNS: dict[str, re.Pattern[str]] = {
    "override_previous_instructions": re.compile(
        r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|any)\b"
        r"[^.\n]{0,30}\b(instructions?|rules?|prompts?|guidelines?|policy|policies)\b",
        re.I,
    ),
    "override_verb_and_object": re.compile(
        rf"\b{_OVERRIDE_VERBS}\b[^.\n]{{0,60}}\b{_OVERRIDE_OBJECTS}\b", re.I
    ),
    "role_reassignment": re.compile(
        rf"\byou are (?:now|no longer)\b[^.\n]{{0,25}}\b{_AI_TERMS}\b"
        rf"|\bact as (?:the|a|an)\b[^.\n]{{0,20}}\b{_AI_TERMS}\b|\bpretend to be\b"
        r"|\b(?:your|the) new (?:role|task|job|instructions?)\b|\bnew (?:system )?(?:instructions?|task)\b"
        r"|\bfrom now on\b[^.\n]{0,30}\byou (?:will|must|shall|should)\b"
        r"|^\s*(?:assistant|ai|model|llm)\s*[,:]|\bnote to the (?:ai|assistant|model)\b",
        re.I,
    ),
    "fake_role_tags": re.compile(
        r"</?\s*(system|assistant|user)\s*>|<\|im_(start|end)\|>|\[/?INST\]"
        r"|(?:^|[\n#>*])\s*system\s*:|#{2,}\s*(?:new|system|instruction)",
        re.I,
    ),
    "prompt_disclosure": re.compile(
        r"\b(reveal|print|show|repeat)\b[^.\n]{0,30}\b(system prompt|instructions)\b", re.I
    ),
    "forced_decision": re.compile(
        r"\b(always|must|should)\b[^.\n]{0,40}\b(mark|set|classify|record|answer|report)\b[^.\n]{0,40}"
        r"\b(clear|approved|no discrepanc\w+|legitimate)\b"
        rf"|\b(?:mark|set|classify|record|answer|respond|output|treat|state|say|approve)\b[^.\n]{{0,50}}\b{_DECISIONS}\b"
        r"|\brecommended[_ ]action\b",
        re.I,
    ),
    "suppress_reporting": re.compile(
        r"\bdo not\b[^.\n]{0,30}\b(flag|escalate|report|review|mention)\b"
        r"|\b(?:do not|don't|must not|mustn't)\s+(?!forget\b|hesitate\b|fail\b)[^.\n]{0,30}"
        r"\b(?:flag|escalate|report|review|mention|raise|alerts?|investigate|check|analy[sz]e)\b"
        r"|\bno need to\b[^.\n]{0,20}\b(?:look|check|review|investigate|escalate|analy[sz]e|flag)\b"
        r"|\bnothing (?:else|further|more|to investigate|to see)\b"
        r"|\bwithout (?:further |any |additional )?(?:scrutiny|checks?|checking|review|verification)\b"
        r"|\bno (?:further|additional) (?:checks?|review|scrutiny|action)\b"
        r"|\bexempt(?:ed)? from\b[^.\n]{0,30}\b(?:monitoring|checks?|review|screening|reconciliation|scrutiny)\b"
        r"|\bstop (?:analy[sz]ing|checking|reviewing|investigating)\b"
        r"|\balready (?:signed|cleared|approved|verified)\b[^.\n]{0,50}\b(?:close|drop|skip|ignore|no need)\b"
        r"|\bjust (?:close|approve|clear|pass)\b",
        re.I,
    ),
    "override_other_languages": re.compile(
        r"\b(?:ignor\w*|oubli\w+|olvid\w+|vergiss\w*|missachte\w*)\b[^.\n]{0,40}\b(?:instructions?|instrucciones|anweisungen|"
        r"regeln|r[eè]gles?|reglas|pr[eé]c[eé]d\w+|anterior\w*|vorherig\w*|todo|tout|alles)\b",
        re.I,
    ),
}

_LEET = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
)
_HYPHEN_LETTERS = re.compile(r"\b(?:[A-Za-z][-._]){2,}[A-Za-z]\b")
_SPACED_LETTERS = re.compile(r"\b(?:[A-Za-z] ){3,}[A-Za-z]\b")
_MIXED_TOKEN = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*[0-9@$])\w{2,12}\b")


def normalise(text: str) -> str:
    """Undo common evasions so the same patterns apply: compatibility forms, invisible characters,
    letter-spacing and digit-for-letter substitution inside words."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = _HYPHEN_LETTERS.sub(lambda m: re.sub(r"[-._]", "", m.group(0)), text)
    text = _SPACED_LETTERS.sub(lambda m: m.group(0).replace(" ", ""), text)
    return _MIXED_TOKEN.sub(lambda m: m.group(0).translate(_LEET), text)


def injection_flags(text: str) -> list[str]:
    candidates = [text]
    normalised = normalise(text)
    if normalised != text:
        candidates.append(normalised)
    return [
        name for name, pattern in PATTERNS.items() if any(pattern.search(c) for c in candidates)
    ]
