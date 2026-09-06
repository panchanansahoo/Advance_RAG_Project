"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """
    Embedding provider abstraction.
    Implementations wrap specific embedding models (local or API-based).
    """

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """
        Generate an embedding for a single query.
        Some models use asymmetric embeddings (different for docs vs queries).

        Args:
            query: The query string to embed.

        Returns:
            A single embedding vector.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        ...
