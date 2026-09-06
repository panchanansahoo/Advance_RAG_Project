"""Abstract base class for document processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.schemas.common import ContentType


@dataclass
class ProcessedContent:
    """
    A single unit of extracted content from a document.
    Processors emit a list of these; the chunker then splits them further.
    """
    text: str
    content_type: ContentType = ContentType.TEXT
    page_number: Optional[int] = None
    section: Optional[str] = None
    table_data: Optional[Dict[str, Any]] = None
    visual_description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseProcessor(ABC):
    """
    Abstract processor that converts a raw file into structured content.

    Each file type (PDF, TXT, CSV, …) gets its own processor subclass.
    """

    @abstractmethod
    async def process(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[ProcessedContent]:
        """
        Process a file and return a list of extracted content units.

        Args:
            file_path: Absolute path to the file on disk.
            metadata: Optional metadata passed from the upload request.

        Returns:
            List of ProcessedContent items ready for chunking.
        """
        ...

    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return the file extensions this processor handles."""
        ...
