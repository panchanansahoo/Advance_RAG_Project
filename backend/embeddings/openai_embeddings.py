"""
OpenAI API-based embedding provider.
"""

from __future__ import annotations

import logging
from typing import List

from backend.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Generate embeddings via the OpenAI Embeddings API."""

    # Default dimensions per model
    _MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-3-small",
        dimension: int | None = None,
    ):
        self._model_name = model_name
        self._dimension = dimension or self._MODEL_DIMS.get(model_name, 1536)
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        client = self._get_client()

        # OpenAI API allows batching up to 2048 texts
        response = await client.embeddings.create(
            input=texts,
            model=self._model_name,
        )
        return [item.embedding for item in response.data]

    async def embed_query(self, query: str) -> List[float]:
        results = await self.embed([query])
        return results[0]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name
