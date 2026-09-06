"""Tests for retrieval components."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from backend.retrieval.retriever import Retriever
from backend.retrieval.vector_store.base import VectorSearchResult


class TestRetriever:
    """Tests for the core Retriever."""

    @pytest.fixture
    def mock_embedding_provider(self):
        provider = AsyncMock()
        provider.embed_query.return_value = [0.1] * 384
        provider.dimension = 384
        return provider

    @pytest.fixture
    def mock_vector_store(self):
        store = AsyncMock()
        store.search.return_value = [
            VectorSearchResult(
                id="chunk-1",
                score=0.95,
                payload={
                    "chunk_id": str(uuid4()),
                    "document_id": str(uuid4()),
                    "content": "Machine learning is a subset of AI.",
                    "content_type": "text",
                    "page_number": 1,
                    "source_filename": "ml.pdf",
                },
            ),
            VectorSearchResult(
                id="chunk-2",
                score=0.87,
                payload={
                    "chunk_id": str(uuid4()),
                    "document_id": str(uuid4()),
                    "content": "Deep learning uses neural networks.",
                    "content_type": "text",
                    "page_number": 2,
                    "source_filename": "ml.pdf",
                },
            ),
        ]
        return store

    @pytest.mark.asyncio
    async def test_retrieve_returns_results(self, mock_embedding_provider, mock_vector_store):
        retriever = Retriever(mock_embedding_provider, mock_vector_store)
        results = await retriever.retrieve("What is machine learning?", top_k=5)

        assert len(results) == 2
        assert results[0].score == 0.95
        mock_embedding_provider.embed_query.assert_called_once_with("What is machine learning?")
        mock_vector_store.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_with_document_filter(self, mock_embedding_provider, mock_vector_store):
        retriever = Retriever(mock_embedding_provider, mock_vector_store)
        doc_id = uuid4()
        await retriever.retrieve("test", top_k=3, document_ids=[doc_id])

        # Check that filters were passed
        call_kwargs = mock_vector_store.search.call_args
        assert call_kwargs.kwargs.get("filters") is not None

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self, mock_embedding_provider, mock_vector_store):
        mock_vector_store.search.return_value = []
        retriever = Retriever(mock_embedding_provider, mock_vector_store)
        results = await retriever.retrieve("obscure query", top_k=5)
        assert len(results) == 0
