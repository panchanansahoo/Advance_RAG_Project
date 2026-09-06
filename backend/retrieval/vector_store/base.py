"""Abstract base class for vector store backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VectorSearchResult:
    """A single result from vector similarity search."""
    id: str
    score: float
    payload: Dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """
    Vector store abstraction for pluggable backends (Qdrant, ChromaDB, etc.).
    """

    @abstractmethod
    async def initialize(self, collection_name: str, dimension: int) -> None:
        """Create or verify the collection/index exists."""
        ...

    @abstractmethod
    async def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors (with optional metadata payloads) to the store."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Perform similarity search and return ranked results."""
        ...

    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """Delete vectors by their IDs."""
        ...

    @abstractmethod
    async def delete_by_filter(self, filters: Dict[str, Any]) -> None:
        """Delete vectors matching the given filters."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return the number of vectors in the store."""
        ...
