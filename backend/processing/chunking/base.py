"""Abstract base class for text chunking strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.schemas.common import ContentType


@dataclass
class ChunkData:
    """A single chunk produced by a chunking strategy."""
    content: str
    chunk_index: int
    content_type: ContentType = ContentType.TEXT
    page_number: Optional[int] = None
    section: Optional[str] = None
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseChunker(ABC):
    """
    Abstract chunker that splits processed content into retrieval-ready chunks.
    """

    @abstractmethod
    def chunk(self, text: str, **kwargs) -> List[ChunkData]:
        """
        Split text into chunks.

        Args:
            text: The text content to chunk.
            **kwargs: Additional context (page_number, section, content_type, etc.)

        Returns:
            List of ChunkData objects.
        """
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Human-readable name of this chunking strategy."""
        ...
