"""
Retrieval service — orchestrates the full Phase 2 retrieval pipeline:

Query Processing → Hybrid Search (Vector + BM25 + RRF) → Reranking → Context Compression

Acts as the bridge between the API layer and all retrieval components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID

from backend.config import get_settings
from backend.embeddings import get_embedding_provider
from backend.retrieval.bm25_search import get_bm25_index
from backend.retrieval.context_compressor import ContextCompressor
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.reranker import get_reranker
from backend.retrieval.vector_store import get_vector_store
from backend.schemas.chunks import RetrievedChunk
from backend.schemas.common import ContentType

logger = logging.getLogger(__name__)


@dataclass
class UnifiedResult:
    """Standardized result container used across retrieval pipeline stages."""
    chunk_id: str = ""
    score: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)
    original_score: float = 0.0
    content: str = ""


class RetrievalService:
    """
    High-level retrieval service used by the query API.

    Phase 2 pipeline:
    1. (Optional) Query rewriting via LLM
    2. Hybrid search: Vector + BM25 with Reciprocal Rank Fusion
    3. Cross-encoder reranking of top candidates
    4. Context compression (dedup, trim, budget)
    5. Return enriched RetrievedChunk objects
    """

    def __init__(self):
        self._hybrid_retriever: HybridRetriever | None = None
        self._compressor: ContextCompressor | None = None

    def _get_hybrid_retriever(self) -> HybridRetriever:
        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetriever(
                embedding_provider=get_embedding_provider(),
                vector_store=get_vector_store(),
            )
        return self._hybrid_retriever

    def _get_compressor(self) -> ContextCompressor:
        if self._compressor is None:
            settings = get_settings()
            self._compressor = ContextCompressor(
                max_tokens=settings.max_context_tokens,
            )
        return self._compressor

    async def retrieve_chunks(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[UUID]] = None,
        owner_key: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve and enrich chunks using the full Phase 2 pipeline.

        Steps:
        1. Query processing (rewriting)
        2. Hybrid retrieval (vector + BM25 + RRF)
        3. Reranking (cross-encoder)
        4. Context compression
        5. Convert to RetrievedChunk objects

        Returns:
            List of RetrievedChunk objects with full metadata for generation.
        """
        settings = get_settings()
        k = top_k or settings.retrieval_top_k

        # ── Step 1: Query Processing ────────────────────────
        effective_query = query
        query_metadata = {}

        if settings.query_rewriting_enabled:
            try:
                from backend.processing.query import QueryProcessor

                processor = QueryProcessor()
                processed = await processor.process(
                    query=query,
                    rewrite=settings.query_rewriting_enabled,
                    expand=settings.query_expansion_enabled,
                    decompose=settings.query_decomposition_enabled,
                )
                effective_query = processed.effective_query
                query_metadata["rewritten_query"] = processed.rewritten
                query_metadata["expanded_terms"] = processed.expanded_terms
            except Exception as e:
                logger.warning("Query processing failed, using original: %s", e)

        # ── Step 2: Hybrid Retrieval ────────────────────────
        retriever = self._get_hybrid_retriever()

        # Fetch more candidates than needed for reranking
        fetch_k = k * 3 if settings.reranker_enabled else k

        hybrid_results = await retriever.retrieve(
            query=effective_query,
            top_k=fetch_k,
            document_ids=document_ids,
            owner_key=owner_key,
        )

        if not hybrid_results:
            logger.info("No results from hybrid retrieval")
            hybrid_results = []

        # ── Step 2.5: Graph Retrieval (Phase 5) ─────────────
        if settings.graph_extraction_enabled and owner_key is None:
            try:
                from backend.retrieval.graph_retriever import GraphRetriever
                graph_retriever = GraphRetriever()
                graph_results = await graph_retriever.search(query=effective_query, top_k=3)
                
                # Fuse graph results into hybrid results as pseudo-chunks
                for gr in graph_results:
                    graph_chunk = UnifiedResult(
                        chunk_id=gr["chunk_id"],
                        score=gr["score"],
                        payload={
                            "chunk_id": gr["chunk_id"],
                            "content": gr["content"],
                            "source_filename": gr["source"],
                            "content_type": "text"
                        },
                        original_score=gr["score"],
                        content=gr["content"],  # Task 4.2: populate content field
                    )
                    hybrid_results.append(graph_chunk)
                    
                if graph_results:
                    logger.info("Added %d graph results to candidates", len(graph_results))
            except Exception as e:
                logger.warning("Graph retrieval failed (non-critical): %s", e)

        if not hybrid_results:
            return []

        logger.info(
            "Hybrid retrieval returned %d candidates (strategy=%s)",
            len(hybrid_results),
            settings.retrieval_strategy,
        )

        # ── Step 3: Reranking ───────────────────────────────
        reranked_results = hybrid_results
        if settings.reranker_enabled and len(hybrid_results) > 1:
            try:
                reranker = get_reranker()
                reranked = await reranker.rerank(
                    query=effective_query,
                    results=hybrid_results,
                    top_k=settings.reranker_top_k,
                )

                # Convert RerankerResult back to a common format
                reranked_results = []
                for r in reranked:
                    content_val = (
                        r.payload.get("content", "")
                        if isinstance(r.payload, dict)
                        else getattr(r, "content", "")
                    )
                    reranked_results.append(UnifiedResult(
                        chunk_id=r.chunk_id,
                        score=r.score,
                        payload=r.payload,
                        original_score=r.original_score,
                        content=content_val,
                    ))

                logger.info("Reranked to %d results", len(reranked_results))
            except Exception as e:
                logger.warning("Reranking failed, using hybrid results: %s", e)

        # ── Step 4: Context Compression ─────────────────────
        final_results = reranked_results
        if settings.context_compression_enabled:
            try:
                compressor = self._get_compressor()
                final_results = compressor.compress(reranked_results)
                logger.info("Compressed to %d chunks", len(final_results))
            except Exception as e:
                logger.warning("Context compression failed: %s", e)

        # ── Step 5: Convert to RetrievedChunks ──────────────
        retrieved_chunks: List[RetrievedChunk] = []
        for result in final_results:
            payload = {}
            score = 0.0

            if hasattr(result, "payload"):
                payload = result.payload
            elif isinstance(result, dict):
                payload = result

            if hasattr(result, "score"):
                score = result.score

            chunk_id_str = payload.get("chunk_id", "")
            if hasattr(result, "chunk_id"):
                chunk_id_str = result.chunk_id

            try:
                retrieved_chunks.append(
                    RetrievedChunk(
                        chunk_id=UUID(chunk_id_str) if chunk_id_str else UUID("00000000-0000-0000-0000-000000000000"),
                        document_id=UUID(payload.get("document_id", "00000000-0000-0000-0000-000000000000")),
                        content=payload.get("content", ""),
                        content_type=ContentType(payload.get("content_type", "text")),
                        page_number=payload.get("page_number"),
                        section=payload.get("section"),
                        score=score,
                        source_filename=payload.get("source_filename"),
                        metadata={
                            k: v for k, v in payload.items()
                            if k not in {
                                "content", "chunk_id", "document_id",
                                "content_type", "page_number", "section",
                                "source_filename",
                            }
                        },
                    )
                )
            except Exception as e:
                logger.warning("Failed to convert result to chunk: %s", e)

        logger.info(
            "Retrieval service returned %d chunks (pipeline: query_rewrite=%s, hybrid=%s, rerank=%s, compress=%s)",
            len(retrieved_chunks),
            settings.query_rewriting_enabled,
            settings.retrieval_strategy,
            settings.reranker_enabled,
            settings.context_compression_enabled,
        )

        return retrieved_chunks

    async def build_bm25_index(self) -> None:
        """
        Build/rebuild the BM25 index from all chunks in the vector store.
        Called after document ingestion.
        """
        try:
            from backend.database.connection import _get_session_factory
            from backend.database.repositories import ChunkRepository
            from sqlalchemy import select
            from backend.database.models import Chunk as ChunkModel

            session = _get_session_factory()()
            try:
                result = await session.execute(
                    select(ChunkModel).order_by(ChunkModel.chunk_index)
                )
                db_chunks = list(result.scalars().all())

                bm25_chunks = []
                for chunk in db_chunks:
                    bm25_chunks.append({
                        "chunk_id": str(chunk.id),
                        "document_id": str(chunk.document_id),
                        "content": chunk.content,
                        "content_type": chunk.content_type.value if hasattr(chunk.content_type, 'value') else str(chunk.content_type),
                        "page_number": chunk.page_number,
                        "section": chunk.section,
                        "source_filename": (chunk.metadata_ or {}).get("source_filename", ""),
                    })

                bm25_index = get_bm25_index()
                bm25_index.build_index(bm25_chunks)
                logger.info("BM25 index built with %d chunks", len(bm25_chunks))
            finally:
                await session.close()

        except Exception as e:
            logger.warning("Failed to build BM25 index: %s", e)


# ── Singleton accessor (Task 4.1) ───────────────────────────
_retrieval_service: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    """Return a cached RetrievalService singleton."""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service

