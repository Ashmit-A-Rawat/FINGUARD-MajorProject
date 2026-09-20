"""Text cleaning. Removes content that can hide instructions from a human reader."""

import re
import unicodedata
from dataclasses import dataclass, field

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# zero-width and bidirectional-override characters: invisible to a reader, visible to a model
_INVISIBLE = re.compile("[​‌‍⁠﻿‪-‮⁦-⁩]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class CleanResult:
    text: str
    actions: dict[str, int] = field(default_factory=dict)


def clean_text(text: str) -> CleanResult:
    """NFC-normalise, drop hidden/control content, tidy whitespace. Every removal is counted."""
    actions: dict[str, int] = {}

    def sub(pattern: re.Pattern[str], name: str, current: str) -> str:
        cleaned, n = pattern.subn("", current)
        if n:
            actions[name] = n
        return cleaned

    out = unicodedata.normalize("NFC", text)
    out = sub(_HTML_COMMENT, "html_comment_removed", out)
    out = sub(_INVISIBLE, "invisible_char_removed", out)
    out = sub(_CONTROL, "control_char_removed", out)
    out = "\n".join(line.rstrip() for line in out.splitlines())
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return CleanResult(out, actions)
