# LLM Architecture

Status: Phase 7 (knowledge base and retrieval) documented below; Phase 8-10 (LLM provider, RAG prompts, guardrails) are still to come. Will document the LLMProvider abstraction (Mock/Qwen/Mistral), structured output validation, RAG prompt design (FACTS / INFERENCES / RECOMMENDATIONS) and prompt-injection defenses.


# Knowledge base and retrieval (Phase 7, `knowledge_base/`)

No LLM is involved yet. This layer only turns reference documents into retrievable, attributable chunks.

```
files -> safe load (extension, size, UTF-8, path, front matter) -> clean (hidden content removed, counted)
      -> sections (## headings; preamble kept, not indexed) -> chunks (sentence-aware, overlap)
      -> injection tripwire flags + trust label -> embeddings (MiniLM) -> Chroma + BM25 -> retriever
```

## Documents
`knowledge_base/documents/` holds 12 documents (about 3,500 words) that I wrote for this project. **Every one is
labelled SYNTHETIC POLICY: invented, not real regulation, legal advice or any bank's procedure**, and numeric
thresholds are illustrative. They describe how FIN-GUARD's own engines work (KYC, anomaly types, reconciliation rules
REC-001..010, decision states, sign-off, audit, privacy, AI-reporting standards). Real regulatory text should only be added
if you supply it.

## Safe handling
- Only `.md`/`.txt`, at most 1 MB, strict UTF-8, no path outside the source root (symlinks resolved and checked),
  required front matter, unique document ids. A bad file is **rejected with a reason** and reported, never silently skipped.
- Cleaning removes HTML comments, zero-width/bidirectional characters and control characters (places to hide text from
  a human reader) and **counts** each removal on the document.
- **Trust is a property of the source directory, not of the file's content.** `SourceSpec(path, trust="trusted"|"untrusted")`;
  a file cannot claim to be trusted in its own front matter (tested).

## Retrieval
Modes: `bm25` (own BM25, normalised to [0,1]), `dense` (ChromaDB cosine over MiniLM embeddings), `hybrid` (reciprocal
rank fusion, k=60). `search()` returns `RetrievedChunk`: `document_id`, `text`, `score`, `rank`, `metadata`
(`document_id`, `source`, `section`, `section_id`, `page`, `version`, `trust`, `injection_flags`) and per-component scores.
`page` is `None` for Markdown/text sources. An index records its embedding model, chunking config and document hashes in
`manifest.json`; opening it with a different embedder raises `IndexMismatchError`.

## Prompt-injection stance (retrieved text is DATA)
1. **Untrusted chunks are excluded by default** (`include_untrusted=False`). A poisoning experiment showed that ranking gives no
   protection: topically matching untrusted documents reached the top-5 for 6/6 adversarial queries and were top-1 for 5-6 of 6
   (see EXP-RAG-01).
2. A regex **tripwire** flags instruction-like text (override / role reassignment / fake role tags / forced decisions /
   suppressing escalation). It flags, never deletes, and is *not* a defence: it cannot see plain misinformation and can be evaded
   by rephrasing.
3. The real defences come in Phases 9-10: retrieved text is passed to the model as fenced quoted data with source ids, never
   as instructions; claims must cite evidence; unsupported claims route to REVIEW.
4. **Not solved here:** a poisoned document inside a *trusted* source. Only provenance, review of the knowledge base, and the
   Phase 10 evidence validation can help.

## Limitations
- Single-author corpus and question set; chunk-size experiments are degenerate (all sections are shorter than 120 words).
- One embedding model (MiniLM), no fine-tuning, no cross-encoder reranking.
- Markdown/text only: no PDF or Word extraction (would need its own safety review).
