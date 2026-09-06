"""
Semantic chunking — groups sentences by embedding similarity.

Groups consecutive sentences together while they remain semantically similar.
When similarity drops below a threshold, a new chunk boundary is created.
Falls back to sentence chunking if the embedding provider is unavailable.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from backend.processing.chunking.base import BaseChunker, ChunkData
from backend.schemas.common import ContentType

logger = logging.getLogger(__name__)


class SemanticChunker(BaseChunker):
    """
    Split text into chunks based on semantic similarity between sentences.

    Algorithm:
    1. Sentence-tokenize the text
    2. Embed each sentence
    3. Compute cosine similarity between consecutive sentences
    4. Place chunk boundaries where similarity drops below threshold
    5. Merge very small chunks into neighbors
    """

    def __init__(
        self,
        chunk_size: int = 512,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        chunk_overlap: int = 0,
    ):
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def strategy_name(self) -> str:
        return "semantic"

    def chunk(self, text: str, **kwargs) -> List[ChunkData]:
        if not text or not text.strip():
            return []

        page_number = kwargs.get("page_number")
        section = kwargs.get("section")
        content_type = kwargs.get("content_type", ContentType.TEXT)
        start_index = kwargs.get("start_index", 0)

        # Tokenize into sentences
        sentences = self._sentence_tokenize(text)
        if not sentences:
            return []

        # If only a few sentences, return as single chunk
        if len(sentences) <= 3:
            full_text = " ".join(sentences).strip()
            return [
                ChunkData(
                    content=full_text,
                    chunk_index=start_index,
                    content_type=content_type,
                    page_number=page_number,
                    section=section,
                    token_count=len(full_text.split()),
                )
            ] if full_text else []

        # Try to get embeddings for semantic splitting
        try:
            embeddings = self._get_sentence_embeddings(sentences)
            boundaries = self._find_boundaries(embeddings)
        except Exception as e:
            logger.warning(
                "Semantic chunking failed, falling back to character-based grouping: %s", e
            )
            boundaries = self._fallback_boundaries(sentences)

        # Group sentences by boundaries
        groups = self._group_sentences(sentences, boundaries)

        # Build ChunkData objects
        chunks: List[ChunkData] = []
        idx = start_index
        for group in groups:
            chunk_text = " ".join(group).strip()
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

        return chunks

    def _sentence_tokenize(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        try:
            import nltk
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
            from nltk.tokenize import sent_tokenize
            return [s.strip() for s in sent_tokenize(text.strip()) if s.strip()]
        except ImportError:
            # Fallback: split on period + space
            import re
            parts = re.split(r'(?<=[.!?])\s+', text.strip())
            return [p.strip() for p in parts if p.strip()]

    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Get embeddings for sentences using the configured embedding provider."""
        from backend.embeddings import get_embedding_provider
        import asyncio

        provider = get_embedding_provider()

        # We need to run async embed in sync context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context — use a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, provider.embed(sentences))
                vectors = future.result()
        else:
            vectors = asyncio.run(provider.embed(sentences))

        return np.array(vectors)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _find_boundaries(self, embeddings: np.ndarray) -> List[int]:
        """
        Find chunk boundary indices where consecutive similarity drops
        below the threshold.
        """
        boundaries = []
        for i in range(1, len(embeddings)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < self.similarity_threshold:
                boundaries.append(i)
        return boundaries

    def _fallback_boundaries(self, sentences: List[str]) -> List[int]:
        """Create boundaries based on character count (fallback when embeddings fail)."""
        boundaries = []
        current_length = 0
        for i, sentence in enumerate(sentences):
            current_length += len(sentence)
            if current_length >= self.chunk_size and i > 0:
                boundaries.append(i)
                current_length = 0
        return boundaries

    def _group_sentences(
        self, sentences: List[str], boundaries: List[int]
    ) -> List[List[str]]:
        """Group sentences into chunks based on boundary indices, merging small chunks."""
        groups: List[List[str]] = []
        prev = 0
        for boundary in sorted(boundaries):
            group = sentences[prev:boundary]
            if group:
                groups.append(group)
            prev = boundary
        # Last group
        remaining = sentences[prev:]
        if remaining:
            groups.append(remaining)

        # Merge very small groups into their neighbors
        merged: List[List[str]] = []
        for group in groups:
            group_len = sum(len(s) for s in group)
            if merged and group_len < self.min_chunk_size:
                merged[-1].extend(group)
            else:
                merged.append(group)

        return merged if merged else [sentences]
