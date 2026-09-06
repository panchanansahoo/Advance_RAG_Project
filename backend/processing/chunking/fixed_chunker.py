"""
Fixed-size chunking with configurable overlap — the Phase 1 baseline strategy.
"""

from __future__ import annotations

from typing import List

from backend.processing.chunking.base import BaseChunker, ChunkData
from backend.schemas.common import ContentType


class FixedChunker(BaseChunker):
    """
    Split text into fixed-size character chunks with overlap.

    This is the simplest chunking strategy and serves as the baseline.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def strategy_name(self) -> str:
        return "fixed"

    def chunk(self, text: str, **kwargs) -> List[ChunkData]:
        if not text or not text.strip():
            return []

        page_number = kwargs.get("page_number")
        section = kwargs.get("section")
        content_type = kwargs.get("content_type", ContentType.TEXT)
        start_index = kwargs.get("start_index", 0)

        chunks: List[ChunkData] = []
        text = text.strip()
        start = 0
        idx = start_index

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at a sentence boundary or whitespace
            if end < len(text):
                # Look backwards for a good break point
                break_point = self._find_break_point(text, start, end)
                if break_point > start:
                    end = break_point

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    ChunkData(
                        content=chunk_text,
                        chunk_index=idx,
                        content_type=content_type,
                        page_number=page_number,
                        section=section,
                        token_count=len(chunk_text.split()),
                    )
                )
                idx += 1

            # Move start forward by (chunk_size - overlap)
            step = max(self.chunk_size - self.chunk_overlap, 1)
            start += step

        return chunks

    @staticmethod
    def _find_break_point(text: str, start: int, end: int) -> int:
        """
        Find a clean break point near `end`, preferring sentence boundaries.
        """
        # Prefer sentence-ending punctuation
        for sep in [". ", ".\n", "? ", "!\n", "?\n", "!\n", "\n\n", "\n"]:
            pos = text.rfind(sep, start + 100, end)
            if pos != -1:
                return pos + len(sep)

        # Fall back to any whitespace
        pos = text.rfind(" ", start + 100, end)
        if pos != -1:
            return pos + 1

        return end
