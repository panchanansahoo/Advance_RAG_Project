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

    # In production (e.g. Render Free tier with 512MB RAM), sentence_transformer causes
    # Out of Memory crashes. Auto-switch to lightweight cloud API embeddings if available.
    if provider == "sentence_transformer" and settings.app_env.lower() == "production":
        google_key = settings.google_api_key.get_secret_value() if settings.google_api_key else ""
        openai_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
        placeholders = {"your-openai-api-key-here", "your-google-api-key-here", "", "changeme"}

        if google_key and google_key.lower() not in placeholders:
            logger.info("Production environment detected: using Gemini embeddings to conserve RAM.")
            provider = "gemini"
        elif openai_key and openai_key.lower() not in placeholders:
            logger.info("Production environment detected: using OpenAI embeddings to conserve RAM.")
            provider = "openai"

    if provider == "openai":
        from backend.embeddings.openai_embeddings import OpenAIEmbeddingProvider

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        model_name = settings.embedding_model if (settings.embedding_model or "").startswith("text-embedding") else "text-embedding-3-small"
        dimension = settings.embedding_dimension if settings.embedding_dimension in (1536, 3072) else 1536
        _instance = OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model_name=model_name,
            dimension=dimension,
        )
    elif provider in ("gemini", "google"):
        from backend.embeddings.gemini_embeddings import GeminiEmbeddingProvider

        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini embeddings")
        raw_model = (settings.embedding_model or "").lower()
        if "gemini-embedding" in raw_model:
            model_name = "models/gemini-embedding-001"
            dim = 3072
        elif "embedding" in raw_model and not raw_model.startswith("all-"):
            model_name = settings.embedding_model if settings.embedding_model.startswith("models/") else f"models/{settings.embedding_model}"
            dim = 3072 if "gemini-embedding" in model_name else (settings.embedding_dimension or 768)
        else:
            model_name = "models/gemini-embedding-001"
            dim = 3072

        _instance = GeminiEmbeddingProvider(
            api_key=settings.google_api_key,
            model_name=model_name,
            dimension=dim,
        )
    else:  # default: sentence_transformer
        from backend.embeddings.sentence_transformer import SentenceTransformerProvider

        model_name = settings.embedding_model
        dimension = settings.embedding_dimension
        # Prevent 438MB all-mpnet-base-v2 on 512MB RAM servers
        if model_name == "all-mpnet-base-v2" and settings.app_env.lower() == "production":
            logger.warning("Replacing heavy 'all-mpnet-base-v2' with lightweight 'all-MiniLM-L6-v2' to prevent OOM crash.")
            model_name = "all-MiniLM-L6-v2"
            dimension = 384

        _instance = SentenceTransformerProvider(
            model_name=model_name,
            dimension=dimension,
        )

    logger.info(
        "Embedding provider: %s (model=%s, dim=%d)",
        provider,
        _instance.model_name,
        _instance.dimension,
    )
    return _instance
