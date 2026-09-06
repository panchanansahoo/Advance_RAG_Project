"""
Context Compression — reduces redundancy and irrelevance in retrieved
chunks before sending them to the LLM.

Phase 2 component implementing PRD §13:
• Deduplicate retrieved content
• Remove irrelevant information
• Extract supporting evidence
• Send the smallest useful evidence set to the LLM
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


class ContextCompressor:
    """
    Compresses the set of retrieved chunks to reduce token usage
    and improve LLM answer quality by removing noise.
    """

    def __init__(self, max_tokens: int = 3000, similarity_threshold: float = 0.85):
        """
        Args:
            max_tokens: Maximum approximate tokens for the compressed context.
            similarity_threshold: Jaccard similarity threshold for deduplication.
        """
        self.max_tokens = max_tokens
        self.similarity_threshold = similarity_threshold

    def compress(self, chunks: List[Any]) -> List[Any]:
        """
        Compress a list of retrieved chunks.

        Steps:
        1. Deduplicate near-identical chunks
        2. Trim excessively long chunks
        3. Enforce total token budget
        4. Return compressed chunk list (preserving order by relevance)

        Args:
            chunks: List of chunk objects with .content and .score attributes.

        Returns:
            Filtered and compressed list of chunks.
        """
        if not chunks:
            return []

        # Step 1: Deduplicate
        deduped = self._deduplicate(chunks)
        logger.debug("Dedup: %d → %d chunks", len(chunks), len(deduped))

        # Step 2: Trim individual chunks
        trimmed = [self._trim_chunk(c) for c in deduped]

        # Step 3: Enforce token budget
        budgeted = self._enforce_token_budget(trimmed)
        logger.info(
            "Context compressed: %d → %d chunks (budget: %d tokens)",
            len(chunks),
            len(budgeted),
            self.max_tokens,
        )

        return budgeted

    def _deduplicate(self, chunks: List[Any]) -> List[Any]:
        """Remove near-duplicate chunks using Jaccard similarity on token sets."""
        if len(chunks) <= 1:
            return chunks

        seen_token_sets: List[Set[str]] = []
        unique_chunks = []

        for chunk in chunks:
            content = self._get_content(chunk)
            tokens = set(content.lower().split())

            is_duplicate = False
            for seen_set in seen_token_sets:
                # Jaccard similarity
                if not tokens or not seen_set:
                    continue
                intersection = tokens & seen_set
                union = tokens | seen_set
                similarity = len(intersection) / len(union) if union else 0

                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_token_sets.append(tokens)
                unique_chunks.append(chunk)

        return unique_chunks

    def _trim_chunk(self, chunk: Any, max_chunk_tokens: int = 500) -> Any:
        """Trim excessively long chunks while preserving complete sentences."""
        content = self._get_content(chunk)
        words = content.split()

        if len(words) <= max_chunk_tokens:
            return chunk

        # Trim to max tokens, then find the last sentence boundary
        trimmed = " ".join(words[:max_chunk_tokens])
        # Try to end at a sentence boundary
        last_period = trimmed.rfind(". ")
        if last_period > len(trimmed) * 0.5:
            trimmed = trimmed[: last_period + 1]

        self._set_content(chunk, trimmed)
        return chunk

    def _enforce_token_budget(self, chunks: List[Any]) -> List[Any]:
        """Keep adding chunks (by relevance order) until the budget is exceeded."""
        result = []
        total_tokens = 0

        for chunk in chunks:
            content = self._get_content(chunk)
            chunk_tokens = len(content.split())

            if total_tokens + chunk_tokens > self.max_tokens and result:
                break

            result.append(chunk)
            total_tokens += chunk_tokens

        return result

    @staticmethod
    def _get_content(chunk: Any) -> str:
        """Extract text content from various chunk types."""
        if hasattr(chunk, "content") and chunk.content:
            return chunk.content
        if hasattr(chunk, "payload") and isinstance(chunk.payload, dict):
            return chunk.payload.get("content", "")
        if isinstance(chunk, dict):
            return chunk.get("content", "")
        if hasattr(chunk, "content"):
            return chunk.content or ""
        return str(chunk)

    @staticmethod
    def _set_content(chunk: Any, content: str) -> None:
        """Set text content on various chunk types."""
        if hasattr(chunk, "content"):
            chunk.content = content
        elif hasattr(chunk, "payload") and isinstance(chunk.payload, dict):
            chunk.payload["content"] = content
        elif isinstance(chunk, dict):
            chunk["content"] = content
