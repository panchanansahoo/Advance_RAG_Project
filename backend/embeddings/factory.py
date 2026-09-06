"""Factory for creating the configured embedding provider."""

from __future__ import annotations

import logging

from backend.config import get_settings
from backend.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)

_instance: BaseEmbeddingProvider | None = None


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Return a cached embedding provider based on settings."""
    global _instance
    if _instance is not None:
        return _instance

    settings = get_settings()
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        from backend.embeddings.openai_embeddings import OpenAIEmbeddingProvider

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        _instance = OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
        )
    else:  # default: sentence_transformer
        from backend.embeddings.sentence_transformer import SentenceTransformerProvider

        _instance = SentenceTransformerProvider(
            model_name=settings.embedding_model,
        )

    logger.info(
        "Embedding provider: %s (model=%s, dim=%d)",
        provider,
        _instance.model_name,
        _instance.dimension,
    )
    return _instance
