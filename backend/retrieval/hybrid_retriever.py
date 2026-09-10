"""
Hybrid retriever combining Vector Search + BM25 with Reciprocal Rank Fusion.

Phase 2 core component: merges semantic similarity (vector) with
exact keyword matching (BM25) for better recall.

RRF formula: score(d) = Σ  weight_i / (k + rank_i(d))
where k is a constant (default 60) and rank_i is the rank in source i.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from backend.config import get_settings
from backend.embeddings.base import BaseEmbeddingProvider
from backend.retrieval.bm25_search import BM25SearchResult, get_bm25_index
from backend.retrieval.vector_store.base import BaseVectorStore, VectorSearchResult

logger = logging.getLogger(__name__)


class HybridSearchResult:
    """A result from hybrid search with fused score."""

    def __init__(
        self,
        chunk_id: str,
        score: float,
        payload: Dict[str, Any],
        vector_score: float = 0.0,
        bm25_score: float = 0.0,
        vector_rank: int = -1,
        bm25_rank: int = -1,
    ):
        self.chunk_id = chunk_id
        self.score = score
        self.payload = payload
        self.vector_score = vector_score
        self.bm25_score = bm25_score
        self.vector_rank = vector_rank
        self.bm25_rank = bm25_rank

    def __repr__(self):
        return (
            f"HybridSearchResult(chunk_id={self.chunk_id}, "
            f"score={self.score:.4f}, vs={self.vector_score:.4f}, bm25={self.bm25_score:.4f})"
        )


class HybridRetriever:
    """
    Combines vector similarity search with BM25 keyword search
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.settings = get_settings()

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[UUID]] = None,
        owner_key: Optional[str] = None,
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid retrieval with RRF fusion.

        Steps:
        1. Run vector search
        2. Run BM25 search (if enabled and index is built)
        3. Fuse results using RRF
        4. Return top-K fused results

        Args:
            query: The search query.
            top_k: Number of results to return after fusion.
            document_ids: Optional document filter.

        Returns:
            List of HybridSearchResult sorted by fused score.
        """
        strategy = self.settings.retrieval_strategy.lower()
        doc_id_strs = [str(d) for d in document_ids] if document_ids else None

        vector_results: List[VectorSearchResult] = []
        bm25_results: List[BM25SearchResult] = []

        async def _run_vector():
            nonlocal vector_results
            if strategy in ("vector", "hybrid"):
                query_embedding = await self.embedding_provider.embed_query(query)
                filters = None
                if doc_id_strs:
                    filters = {"document_id": doc_id_strs}
                if owner_key is not None:
                    filters = filters or {}
                    filters["owner_key"] = owner_key
                vector_top_k = max(top_k * 3, self.settings.bm25_top_k)
                vector_results = await self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=vector_top_k,
                    filters=filters,
                )
                logger.info("Vector search returned %d results", len(vector_results))

        async def _run_bm25():
            nonlocal bm25_results
            if strategy in ("bm25", "hybrid") and self.settings.bm25_enabled:
                bm25_index = get_bm25_index()
                if bm25_index.is_built:
                    # BM25 search is CPU-bound, run in thread
                    bm25_results = await asyncio.to_thread(
                        bm25_index.search,
                        query=query,
                        top_k=self.settings.bm25_top_k,
                        document_ids=doc_id_strs,
                        owner_key=owner_key,
                    )
                    logger.info("BM25 search returned %d results", len(bm25_results))
                else:
                    logger.debug("BM25 index not built, skipping keyword search")

        # Run both searches concurrently
        await asyncio.gather(_run_vector(), _run_bm25())

        # ── Fusion ──────────────────────────────────────────
        if strategy == "vector" or not bm25_results:
            # Vector-only: convert to HybridSearchResult
            return [
                HybridSearchResult(
                    chunk_id=r.id,
                    score=r.score,
                    payload=r.payload,
                    vector_score=r.score,
                    vector_rank=i,
                )
                for i, r in enumerate(vector_results[:top_k])
            ]

        if strategy == "bm25" or not vector_results:
            # BM25-only: convert to HybridSearchResult
            return [
                HybridSearchResult(
                    chunk_id=r.chunk_id,
                    score=r.score,
                    payload=r.payload,
                    bm25_score=r.score,
                    bm25_rank=i,
                )
                for i, r in enumerate(bm25_results[:top_k])
            ]

        # ── Reciprocal Rank Fusion ──────────────────────────
        fused = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            vector_weight=self.settings.vector_weight,
            bm25_weight=self.settings.bm25_weight,
            k=self.settings.rrf_k,
        )

        logger.info(
            "RRF fusion: %d vector + %d BM25 → %d unique results",
            len(vector_results),
            len(bm25_results),
            len(fused),
        )

        return fused[:top_k]

    @staticmethod
    def _reciprocal_rank_fusion(
        vector_results: List[VectorSearchResult],
        bm25_results: List[BM25SearchResult],
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        k: int = 60,
    ) -> List[HybridSearchResult]:
        """
        Merge two ranked lists using weighted Reciprocal Rank Fusion.

        RRF(d) = w_v / (k + rank_v(d)) + w_b / (k + rank_b(d))
        """
        # Collect all unique chunk IDs with their metadata
        chunk_data: Dict[str, Dict[str, Any]] = {}

        # Process vector results
        vector_ranks: Dict[str, int] = {}
        vector_scores: Dict[str, float] = {}
        for rank, result in enumerate(vector_results):
            cid = result.payload.get("chunk_id", result.id)
            vector_ranks[cid] = rank
            vector_scores[cid] = result.score
            if cid not in chunk_data:
                chunk_data[cid] = result.payload

        # Process BM25 results
        bm25_ranks: Dict[str, int] = {}
        bm25_scores_map: Dict[str, float] = {}
        for rank, result in enumerate(bm25_results):
            cid = result.chunk_id
            bm25_ranks[cid] = rank
            bm25_scores_map[cid] = result.score
            if cid not in chunk_data:
                chunk_data[cid] = result.payload

        # Compute fused scores
        all_chunk_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        fused_results: List[HybridSearchResult] = []

        for cid in all_chunk_ids:
            rrf_score = 0.0
            v_rank = -1
            b_rank = -1

            if cid in vector_ranks:
                v_rank = vector_ranks[cid]
                rrf_score += vector_weight / (k + v_rank)

            if cid in bm25_ranks:
                b_rank = bm25_ranks[cid]
                rrf_score += bm25_weight / (k + b_rank)

            fused_results.append(
                HybridSearchResult(
                    chunk_id=cid,
                    score=rrf_score,
                    payload=chunk_data.get(cid, {}),
                    vector_score=vector_scores.get(cid, 0.0),
                    bm25_score=bm25_scores_map.get(cid, 0.0),
                    vector_rank=v_rank,
                    bm25_rank=b_rank,
                )
            )

        # Sort by fused score descending
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results
