"""
Local embedding provider using sentence-transformers.
Default model: all-MiniLM-L6-v2 (384 dimensions, fast, good quality).
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import List

from backend.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """
    Generate embeddings using a local sentence-transformers model.
    The model is loaded once and cached.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int | None = None,
    ):
        self._model_name = model_name
        self._model = None
        self._dimension = dimension

    def _load_model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            # Get dimension from a test encode
            test_emb = self._model.encode(["test"])
            self._dimension = len(test_emb[0])
            logger.info(
                "Model loaded: %s (dim=%d)", self._model_name, self._dimension
            )

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._load_model()

        # Run the CPU-intensive encoding in a thread
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                texts, show_progress_bar=False, normalize_embeddings=True
            ),
        )
        return [emb.tolist() for emb in embeddings]

    async def embed_query(self, query: str) -> List[float]:
        results = await self.embed([query])
        return results[0]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name
