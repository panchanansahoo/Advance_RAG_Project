"""Factory for creating the configured vector store backend."""

from __future__ import annotations

import logging

from backend.config import get_settings
from backend.retrieval.vector_store.base import BaseVectorStore

logger = logging.getLogger(__name__)

_instance: BaseVectorStore | None = None


def get_vector_store() -> BaseVectorStore:
    """Return a cached vector store based on settings."""
    global _instance
    if _instance is not None:
        return _instance

    settings = get_settings()
    provider = settings.vector_db_provider.lower()

    if provider == "chroma":
        from backend.retrieval.vector_store.chroma_store import ChromaVectorStore
        _instance = ChromaVectorStore(persist_dir=settings.chroma_persist_dir)
    else:  # default: qdrant
        from backend.retrieval.vector_store.qdrant_store import QdrantVectorStore
        _instance = QdrantVectorStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

    logger.info("Vector store provider: %s", provider)
    return _instance
