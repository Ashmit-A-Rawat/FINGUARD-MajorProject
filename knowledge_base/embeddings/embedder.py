"""Embedding interface for the knowledge base (shared with the KYC engine)."""

from kyc.semantic.embedder import DEFAULT_MODEL, Embedder, SentenceTransformerEmbedder

__all__ = ["DEFAULT_MODEL", "Embedder", "SentenceTransformerEmbedder"]
