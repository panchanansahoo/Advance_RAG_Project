"""
ChromaDB vector store implementation — lightweight alternative for dev/testing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.retrieval.vector_store.base import BaseVectorStore, VectorSearchResult

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Vector store backed by ChromaDB (local, file-persisted)."""

    def __init__(self, persist_dir: str = "./chroma_data"):
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None

    def _get_client(self):
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=chromadb.config.Settings(anonymized_telemetry=False)
            )
            logger.info("ChromaDB initialized at %s", self.persist_dir)
        return self._client

    async def initialize(self, collection_name: str, dimension: int) -> None:
        client = self._get_client()
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready", collection_name)

    async def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if self._collection is None:
            raise RuntimeError("Collection not initialized")

        metadatas = payloads if payloads else [{}] * len(ids)
        # ChromaDB requires string values in metadata
        clean_metadatas = [
            {k: str(v) for k, v in m.items()} for m in metadatas
        ]

        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=clean_metadatas,
            )
            logger.info("Upserted %d vectors to ChromaDB", len(ids))
        except Exception as e:
            if "dimension" in str(e).lower() or "dimensionality" in str(e).lower():
                logger.warning("ChromaDB dimension mismatch (%s). Recreating collection '%s'...", e, self._collection.name)
                client = self._get_client()
                try:
                    client.delete_collection(self._collection.name)
                except Exception:
                    pass
                self._collection = client.get_or_create_collection(
                    name=self._collection.name,
                    metadata={"hnsw:space": "cosine"},
                )
                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=clean_metadatas,
                )
                logger.info("Recreated ChromaDB collection and upserted %d vectors", len(ids))
            else:
                raise

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        if self._collection is None:
            raise RuntimeError("Collection not initialized")

        where_filter = None
        if filters:
            # ChromaDB uses {"key": {"$eq": "value"}} format
            conditions = {}
            for k, v in filters.items():
                conditions[k] = {"$eq": str(v)}
            where_filter = conditions if len(conditions) == 1 else {"$and": [
                {k: v} for k, v in conditions.items()
            ]}

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
            )
        except Exception as e:
            if "dimension" in str(e).lower() or "dimensionality" in str(e).lower():
                logger.warning("ChromaDB search dimension mismatch: %s. Returning empty results.", e)
                return []
            raise

        search_results: List[VectorSearchResult] = []
        if results["ids"] and results["ids"][0]:
            for i, id_ in enumerate(results["ids"][0]):
                score = 1.0  # ChromaDB returns distances; convert if needed
                if results.get("distances") and results["distances"][0]:
                    # Cosine distance → similarity
                    score = 1.0 - results["distances"][0][i]
                payload = {}
                if results.get("metadatas") and results["metadatas"][0]:
                    payload = results["metadatas"][0][i] or {}
                search_results.append(
                    VectorSearchResult(id=id_, score=score, payload=payload)
                )

        return search_results

    async def delete(self, ids: List[str]) -> None:
        if self._collection is None:
            raise RuntimeError("Collection not initialized")
        self._collection.delete(ids=ids)

    async def delete_by_filter(self, filters: Dict[str, Any]) -> None:
        if self._collection is None:
            raise RuntimeError("Collection not initialized")
        where = {k: {"$eq": str(v)} for k, v in filters.items()}
        self._collection.delete(where=where)

    async def count(self) -> int:
        if self._collection is None:
            raise RuntimeError("Collection not initialized")
        return self._collection.count()
