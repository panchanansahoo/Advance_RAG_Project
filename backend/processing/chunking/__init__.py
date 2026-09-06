"""Chunking strategies — factory function to get the right chunker."""

from __future__ import annotations

from backend.processing.chunking.base import BaseChunker, ChunkData
from backend.processing.chunking.fixed_chunker import FixedChunker
from backend.processing.chunking.sentence_chunker import SentenceChunker
from backend.processing.chunking.semantic_chunker import SemanticChunker
from backend.processing.chunking.structure_aware_chunker import StructureAwareChunker


def get_chunker(
    strategy: str = "fixed",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> BaseChunker:
    """Return a chunker instance for the given strategy name."""
    strategy = strategy.lower()
    if strategy == "sentence":
        return SentenceChunker(chunk_size=chunk_size, chunk_overlap=1)
    elif strategy == "semantic":
        return SemanticChunker(chunk_size=chunk_size)
    elif strategy in ("structure_aware", "structure"):
        return StructureAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:  # default: fixed
        return FixedChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


__all__ = [
    "BaseChunker",
    "ChunkData",
    "FixedChunker",
    "SentenceChunker",
    "SemanticChunker",
    "StructureAwareChunker",
    "get_chunker",
]

