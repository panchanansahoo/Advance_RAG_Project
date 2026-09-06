"""
Qdrant vector store implementation — the primary production backend.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from backend.retrieval.vector_store.base import BaseVectorStore, VectorSearchResult

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """Vector store backed by Qdrant."""

    def __init__(self, host: str = "localhost", port: int = 6333):
        self.host = host
        self.port = port
        self._client = None
        self._collection_name = None

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(host=self.host, port=self.port)
            logger.info("Connected to Qdrant at %s:%d", self.host, self.port)
        return self._client

    async def initialize(self, collection_name: str, dimension: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        self._collection_name = collection_name
        client = self._get_client()

        # Check if collection exists
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if not exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension, distance=Distance.COSINE
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d)",
                collection_name,
                dimension,
            )
        else:
            logger.info("Qdrant collection '%s' already exists", collection_name)

    async def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        from qdrant_client.models import PointStruct

        client = self._get_client()
        points = []
        for i, (id_, embedding) in enumerate(zip(ids, embeddings)):
            payload = payloads[i] if payloads else {}
            points.append(
                PointStruct(id=id_, vector=embedding, payload=payload)
            )

        # Batch upsert in groups of 100
        batch_size = 100
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            client.upsert(
                collection_name=self._collection_name,
                points=batch,
            )

        logger.info("Upserted %d vectors to Qdrant", len(points))

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        client = self._get_client()

        # Build Qdrant filter if needed
        qdrant_filter = None
        if filters:
            qdrant_filter = self._build_filter(filters)

        results = client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
        )

        return [
            VectorSearchResult(
                id=str(hit.id),
                score=hit.score,
                payload=hit.payload or {},
            )
            for hit in results
        ]

    async def delete(self, ids: List[str]) -> None:
        from qdrant_client.models import PointIdsList

        client = self._get_client()
        client.delete(
            collection_name=self._collection_name,
            points_selector=PointIdsList(points=ids),
        )
        logger.info("Deleted %d vectors from Qdrant", len(ids))

    async def delete_by_filter(self, filters: Dict[str, Any]) -> None:
        from qdrant_client.models import FilterSelector

        client = self._get_client()
        qdrant_filter = self._build_filter(filters)
        if qdrant_filter:
            client.delete(
                collection_name=self._collection_name,
                points_selector=FilterSelector(filter=qdrant_filter),
            )

    async def count(self) -> int:
        client = self._get_client()
        info = client.get_collection(self._collection_name)
        return info.points_count

    @staticmethod
    def _build_filter(filters: Dict[str, Any]):
        """Convert a simple dict of key-value filters to Qdrant filter format."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                # Match any value in the list
                for v in value:
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=str(v)))
                    )
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=str(value)))
                )

        return Filter(must=conditions) if conditions else None
