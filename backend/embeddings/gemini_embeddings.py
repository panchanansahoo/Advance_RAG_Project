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
        "models/gemini-embedding-001": 3072,
        "gemini-embedding-001": 3072,
        "models/text-embedding-004": 768,
        "text-embedding-004": 768,
        "models/embedding-001": 768,
        "embedding-001": 768,
    }

    def __init__(
        self,
        api_key: str,
        model_name: str = "models/gemini-embedding-001",
        dimension: int | None = None,
    ):
        # Ensure model name has proper prefix if not provided
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        self._model_name = model_name
        if model_name in self._MODEL_DIMS:
            self._dimension = self._MODEL_DIMS[model_name]
        elif dimension is not None and dimension > 0:
            self._dimension = dimension
        else:
            self._dimension = 3072 if "gemini-embedding" in model_name else 768

        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            key = self._api_key.get_secret_value() if hasattr(self._api_key, "get_secret_value") else str(self._api_key)
            self._client = genai.Client(api_key=key)
        return self._client

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        client = self._get_client()

        # Batch in chunks of up to 100 to stay within API limits
        chunk_size = 100
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            try:
                result = await client.aio.models.embed_content(
                    model=self._model_name,
                    contents=chunk,
                )
            except Exception as exc:
                err_str = str(exc).lower()
                if ("not found" in err_str or "404" in err_str or "not supported" in err_str) and self._model_name != "models/gemini-embedding-001":
                    logger.warning(
                        "Model %s failed (%s). Falling back to models/gemini-embedding-001",
                        self._model_name,
                        exc,
                    )
                    self._model_name = "models/gemini-embedding-001"
                    self._dimension = 3072
                    result = await client.aio.models.embed_content(
                        model=self._model_name,
                        contents=chunk,
                    )
                else:
                    raise

            embeddings = [item.values for item in (result.embeddings or [])]
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        client = self._get_client()

        try:
            result = await client.aio.models.embed_content(
                model=self._model_name,
                contents=query,
            )
        except Exception as exc:
            err_str = str(exc).lower()
            if ("not found" in err_str or "404" in err_str or "not supported" in err_str) and self._model_name != "models/gemini-embedding-001":
                logger.warning(
                    "Model %s failed (%s). Falling back to models/gemini-embedding-001",
                    self._model_name,
                    exc,
                )
                self._model_name = "models/gemini-embedding-001"
                self._dimension = 3072
                result = await client.aio.models.embed_content(
                    model=self._model_name,
                    contents=query,
                )
            else:
                raise

        return result.embeddings[0].values if result.embeddings else []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name
