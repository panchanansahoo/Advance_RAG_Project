"""
Cross-Encoder Reranker — rescores retrieved chunks using a more
powerful model for fine-grained relevance assessment.

Phase 2 component: the cross-encoder sees (query, passage) pairs
jointly, producing much better relevance scores than bi-encoder
similarity, at the cost of being slower (hence used only on top-K).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RerankerResult:
    """A reranked result with cross-encoder score."""

    def __init__(
        self,
        chunk_id: str,
        score: float,
        original_score: float,
        payload: Dict[str, Any],
    ):
        self.chunk_id = chunk_id
        self.score = score
        self.original_score = original_score
        self.payload = payload

    def __repr__(self):
        return f"RerankerResult(chunk_id={self.chunk_id}, score={self.score:.4f})"


class CrossEncoderReranker:
    """
    Reranks search results using a cross-encoder model.

    The cross-encoder takes (query, passage) pairs and produces a
    relevance score. This is more accurate than vector similarity
    but too slow for initial retrieval, so we use it as a second-stage
    reranker on top-K candidates.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder reranker: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
            logger.info("Cross-encoder loaded: %s", self._model_name)

    async def rerank(
        self,
        query: str,
        results: List[Any],
        top_k: int = 5,
    ) -> List[RerankerResult]:
        """
        Rerank search results using the cross-encoder.

        Args:
            query: The original user query.
            results: List of search results (must have .chunk_id, .payload, .score).
            top_k: Number of results to return after reranking.

        Returns:
            Reranked list of RerankerResult, sorted by cross-encoder score.
        """
        if not results:
            return []

        self._load_model()

        # Build (query, passage) pairs
        pairs = []
        for result in results:
            content = ""
            if hasattr(result, "payload"):
                content = result.payload.get("content", "")
            elif isinstance(result, dict):
                content = result.get("content", "")

            pairs.append((query, content))

        # Run cross-encoder scoring in a thread (CPU-intensive)
        scores = await asyncio.to_thread(self._model.predict, pairs)

        # Build reranked results
        reranked: List[RerankerResult] = []
        for i, result in enumerate(results):
            chunk_id = ""
            original_score = 0.0
            payload = {}

            if hasattr(result, "chunk_id"):
                chunk_id = result.chunk_id
            elif hasattr(result, "id"):
                chunk_id = result.id

            if hasattr(result, "score"):
                original_score = result.score

            if hasattr(result, "payload"):
                payload = result.payload
            elif isinstance(result, dict):
                payload = result

            reranked.append(
                RerankerResult(
                    chunk_id=chunk_id,
                    score=float(scores[i]),
                    original_score=original_score,
                    payload=payload,
                )
            )

        # Sort by cross-encoder score descending
        reranked.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            "Reranked %d results → top-%d (best=%.4f, worst=%.4f)",
            len(reranked),
            min(top_k, len(reranked)),
            reranked[0].score if reranked else 0,
            reranked[-1].score if reranked else 0,
        )

        return reranked[:top_k]


# ── Singleton ───────────────────────────────────────────────

_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(model_name: Optional[str] = None) -> CrossEncoderReranker:
    """Return a cached reranker instance."""
    global _reranker
    if _reranker is None:
        from backend.config import get_settings
        settings = get_settings()
        _reranker = CrossEncoderReranker(
            model_name=model_name or settings.reranker_model
        )
    return _reranker
