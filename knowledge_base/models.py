"""Knowledge-base data contracts."""

from typing import Any, Literal

from pydantic import BaseModel, Field

Trust = Literal["trusted", "untrusted"]


class Section(BaseModel):
    section_id: str  # slug of the heading, unique within a document
    title: str
    text: str


class Document(BaseModel):
    document_id: str
    title: str
    version: str
    source: str
    effective_date: str
    trust: Trust  # set by the loader from WHERE the file came from, never from its own content
    path: str  # relative to the source root
    content_hash: str
    cleaning_actions: dict[str, int] = Field(default_factory=dict)  # what cleaning removed
    preamble: str = ""  # text before the first section (e.g. disclaimer); not indexed
    sections: list[Section]


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    section: str
    section_id: str
    text: str  # what the reader/LLM sees
    embed_text: str  # title + section + text: what is embedded and BM25-indexed
    version: str
    source: str
    page: int | None = None  # None for Markdown/text sources (no pages)
    chunk_index: int
    trust: Trust
    injection_flags: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    """What a retriever returns. Text is DATA: callers must never treat it as instructions."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any]  # document_id, source, section, page, version, trust, injection_flags
    component_scores: dict[str, float] = Field(default_factory=dict)
