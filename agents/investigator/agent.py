"""Investigation agent. Responsibility: retrieve reference knowledge and ask the LLM for a
structured, ADVISORY investigation. The only agent that calls a language model."""

from agents.context import AgentContext
from agents.contracts import (
    InvestigationInput,
    InvestigationMeta,
    InvestigationResult,
    RetrievalInput,
    RetrievalOutput,
)
from agents.investigator.queries import build_queries
from knowledge_base.models import RetrievedChunk
from llm.prompts.investigation import build_investigation_prompt
from llm.rag.structured import generate_structured
from llm.schemas import InvestigationOutput

MAX_CHUNKS = 4


class InvestigationAgent:
    name = "investigator"

    def __init__(self, ctx: AgentContext, max_chunks: int = MAX_CHUNKS) -> None:
        self._ctx = ctx
        self._max_chunks = max_chunks

    def retrieve(self, inp: RetrievalInput) -> RetrievalOutput:
        """Top trusted chunk per query, de-duplicated, capped to protect the prompt budget."""
        queries = build_queries(inp.evidence)
        chunks: list[RetrievedChunk] = []
        seen: set[str] = set()
        for query in queries:
            for chunk in self._ctx.knowledge_base.retriever.search(query, k=1):  # trusted only
                if chunk.chunk_id not in seen and len(chunks) < self._max_chunks:
                    seen.add(chunk.chunk_id)
                    chunks.append(chunk)
        return RetrievalOutput(queries=queries, chunks=chunks)

    def investigate(self, inp: InvestigationInput) -> InvestigationResult:
        bundle = build_investigation_prompt(inp.evidence, inp.chunks)
        result = generate_structured(self._ctx.llm, bundle.messages, InvestigationOutput)
        first = result.attempts[0].result
        meta = InvestigationMeta(
            prompt_version=bundle.prompt_version,
            nonce=bundle.nonce,
            provider=first.provider,
            model=first.model,
            is_mock=result.is_mock,
            attempts=len(result.attempts),
            errors=[a.error for a in result.attempts if a.error],
            latency_s=result.total_latency_s,
            knowledge_chunk_ids=bundle.knowledge_chunk_ids,
            excluded_suspicious_chunks=bundle.excluded_suspicious,
        )
        return InvestigationResult(investigation=result.value, meta=meta)
