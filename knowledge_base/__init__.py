"""Banking knowledge base: ingestion, chunking, embeddings, vector store, retrieval."""

from backend.app.core.runtime import configure_native_threads

configure_native_threads()  # must run before torch / sentence-transformers are imported
