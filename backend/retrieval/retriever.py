"""
Core retriever: embeds a query and performs vector similarity search.

Phase 1: vector-only search.
Phase 2 will add BM25 + hybrid fusion + reranking.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from backend.embeddings.base import BaseEmbeddingProvider
from backend.retrieval.vector_store.base import BaseVectorStore, VectorSearchResult

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retriever that embeds queries and finds relevant chunks
    from the vector store.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[UUID]] = None,
    ) -> List[VectorSearchResult]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: The user's question.
            top_k: Number of results to return.
            document_ids: Optional filter to search within specific documents.

        Returns:
            Ranked list of VectorSearchResult with IDs, scores, and payloads.
        """
        # 1. Embed the query
        query_embedding = await self.embedding_provider.embed_query(query)

        # 2. Build filters
        filters: Optional[Dict[str, Any]] = None
        if document_ids:
            filters = {"document_id": [str(did) for did in document_ids]}

        # 3. Search the vector store
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        logger.info(
            "Retrieved %d results for query: '%s...'",
            len(results),
            query[:50],
        )
        return results
