"""Safe document loading: extraction of Markdown/text into documents with sections.

Safety rules: only allowed extensions, size cap, strict UTF-8, no path escaping the source root
(symlinks included), required front matter, unique document ids. A bad file is REJECTED WITH A
REASON; it never aborts the load and is never silently skipped.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from knowledge_base.ingestion.cleaning import clean_text
from knowledge_base.models import Document, Section, Trust

ALLOWED_SUFFIXES = {".md", ".txt"}
MAX_BYTES = 1_000_000
REQUIRED_FRONT_MATTER = ("document_id", "title", "version", "source", "effective_date")
_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)


@dataclass
class Rejected:
    path: str
    reason: str


@dataclass
class LoadResult:
    documents: list[Document] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "section"


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(raw)
    if match is None:
        raise ValueError("missing front matter block")
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            raise ValueError(f"malformed front matter line: {line!r}")
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    missing = [k for k in REQUIRED_FRONT_MATTER if not meta.get(k)]
    if missing:
        raise ValueError(f"front matter missing: {', '.join(missing)}")
    return meta, match.group(2)


def _split_sections(body: str) -> tuple[str, list[Section]]:
    """Return (preamble, sections). Text before the first '## ' heading is the PREAMBLE
    (for example a disclaimer): kept on the document but not indexed as a searchable chunk."""
    sections: list[Section] = []
    used: set[str] = set()
    preamble_lines: list[str] = []
    title: str | None = None
    lines: list[str] = []

    def flush() -> None:
        text = "\n".join(lines).strip()
        if title is not None and text:
            base = slugify(title)
            section_id, n = base, 2
            while section_id in used:
                section_id, n = f"{base}-{n}", n + 1
            used.add(section_id)
            sections.append(Section(section_id=section_id, title=title, text=text))

    for line in body.splitlines():
        if line.startswith("## "):
            flush()
            title, lines = line[3:].strip(), []
        elif line.startswith("# "):  # H1 is the document title, already in front matter
            continue
        elif title is None:
            preamble_lines.append(line)
        else:
            lines.append(line)
    flush()
    return "\n".join(preamble_lines).strip(), sections


def load_documents(root: Path, trust: Trust = "trusted") -> LoadResult:
    """Load every allowed file under ``root``. ``trust`` describes the SOURCE of this directory."""
    result = LoadResult()
    root_resolved = root.resolve()
    seen_ids: set[str] = set()
    for path in sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink()):
        rel = str(path.relative_to(root))
        resolved = path.resolve()
        if not resolved.is_relative_to(root_resolved):
            result.rejected.append(Rejected(rel, "outside_source_root"))
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            result.rejected.append(Rejected(rel, f"unsupported_extension:{path.suffix or 'none'}"))
            continue
        if resolved.stat().st_size > MAX_BYTES:
            result.rejected.append(Rejected(rel, "file_too_large"))
            continue
        try:
            raw = resolved.read_bytes().decode("utf-8")  # strict: invalid bytes are an error
            meta, body = _parse_front_matter(raw.replace("\r\n", "\n"))
            cleaned = clean_text(body)
            preamble, sections = _split_sections(cleaned.text)
        except (UnicodeDecodeError, ValueError) as exc:
            result.rejected.append(Rejected(rel, f"invalid_document:{exc}"))
            continue
        if not sections:
            result.rejected.append(Rejected(rel, "no_content"))
            continue
        if meta["document_id"] in seen_ids:
            result.rejected.append(Rejected(rel, f"duplicate_document_id:{meta['document_id']}"))
            continue
        seen_ids.add(meta["document_id"])
        result.documents.append(
            Document(
                document_id=meta["document_id"],
                title=meta["title"],
                version=meta["version"],
                source=meta["source"],
                effective_date=meta["effective_date"],
                trust=trust,
                path=rel,
                content_hash=hashlib.sha256(raw.encode()).hexdigest(),
                cleaning_actions=cleaned.actions,
                preamble=preamble,
                sections=sections,
            )
        )
    return result
