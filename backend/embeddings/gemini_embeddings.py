"""
Google Gemini API-based embedding provider.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from backend.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Generate embeddings via the Google Gemini Embeddings API."""

    _MODEL_DIMS = {
        "models/text-embedding-004": 768,
        "text-embedding-004": 768,
        "models/embedding-001": 768,
        "embedding-001": 768,
    }

    def __init__(
        self,
        api_key: str,
        model_name: str = "models/text-embedding-004",
        dimension: int | None = None,
    ):
        # Ensure model name has proper prefix if not provided
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        self._model_name = model_name
        self._dimension = dimension or self._MODEL_DIMS.get(model_name, 768)
        self._api_key = api_key
        self._configured = False

    def _ensure_configured(self):
        if not self._configured:
            import google.generativeai as genai
            key = self._api_key.get_secret_value() if hasattr(self._api_key, "get_secret_value") else str(self._api_key)
            genai.configure(api_key=key)
            self._configured = True

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        self._ensure_configured()
        import google.generativeai as genai

        # Batch in chunks of up to 100 to stay within API limits
        chunk_size = 100
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            result = await genai.embed_content_async(
                model=self._model_name,
                content=chunk,
                task_type="retrieval_document",
            )
            embeddings = result.get("embedding", [])
            # If a single item was returned, ensure it's a list of lists
            if embeddings and isinstance(embeddings[0], float):
                embeddings = [embeddings]
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        self._ensure_configured()
        import google.generativeai as genai

        result = await genai.embed_content_async(
            model=self._model_name,
            content=query,
            task_type="retrieval_query",
        )
        return result.get("embedding", [])

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name
