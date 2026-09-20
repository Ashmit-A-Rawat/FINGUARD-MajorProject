"""Section-aware chunking: never crosses a section boundary; splits long sections by sentence."""

import re

from pydantic import BaseModel, Field

from knowledge_base.ingestion.sanitize import injection_flags
from knowledge_base.models import Chunk, Document

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class ChunkingConfig(BaseModel):
    max_words: int = Field(120, ge=10)
    overlap_words: int = Field(20, ge=0)
    min_words: int = Field(15, ge=1)  # a shorter trailing chunk is merged into the previous one


def _units(text: str) -> list[str]:
    """Sentences, keeping list items and paragraphs as separate units."""
    units: list[str] = []
    for line in (ln.strip() for ln in text.splitlines()):
        if line:
            units.extend(s for s in _SENTENCE_END.split(line) if s)
    return units


def _words(text: str) -> int:
    return len(text.split())


def split_section(text: str, config: ChunkingConfig) -> list[str]:
    units = _units(text)
    chunks: list[list[str]] = []
    current: list[str] = []
    for unit in units:
        if current and _words(" ".join(current)) + _words(unit) > config.max_words:
            chunks.append(current)
            carried: list[str] = []
            for previous in reversed(current):  # carry a sentence tail as overlap
                if _words(" ".join([previous, *carried])) > config.overlap_words:
                    break
                carried.insert(0, previous)
            current = carried
        current.append(unit)
    if current:
        chunks.append(current)
    if len(chunks) > 1 and _words(" ".join(chunks[-1])) < config.min_words:
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + [u for u in tail if u not in chunks[-1]]
    return [" ".join(c) for c in chunks]


def chunk_document(document: Document, config: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in document.sections:
        for index, text in enumerate(split_section(section.text, config)):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}:{section.section_id}:{index}",
                    document_id=document.document_id,
                    title=document.title,
                    section=section.title,
                    section_id=section.section_id,
                    text=text,
                    embed_text=f"{document.title} - {section.title}. {text}",
                    version=document.version,
                    source=document.source,
                    chunk_index=index,
                    trust=document.trust,
                    injection_flags=injection_flags(text),
                )
            )
    return chunks
