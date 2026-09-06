"""Tests for hybrid retrieval and RRF fusion."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from backend.retrieval.hybrid_retriever import HybridRetriever, HybridSearchResult
from backend.retrieval.vector_store.base import VectorSearchResult
from backend.retrieval.bm25_search import BM25SearchResult


class TestRRFFusion:
    """Tests for the Reciprocal Rank Fusion algorithm."""

    def test_rrf_basic_fusion(self):
        """Test that RRF correctly merges two ranked lists."""
        vector_results = [
            VectorSearchResult(id="v1", score=0.95, payload={"chunk_id": "c1", "content": "ML"}),
            VectorSearchResult(id="v2", score=0.85, payload={"chunk_id": "c2", "content": "DL"}),
            VectorSearchResult(id="v3", score=0.75, payload={"chunk_id": "c3", "content": "NLP"}),
        ]

        bm25_results = [
            BM25SearchResult(chunk_id="c2", score=8.5, payload={"chunk_id": "c2", "content": "DL"}),
            BM25SearchResult(chunk_id="c4", score=7.2, payload={"chunk_id": "c4", "content": "CV"}),
            BM25SearchResult(chunk_id="c1", score=5.1, payload={"chunk_id": "c1", "content": "ML"}),
        ]

        fused = HybridRetriever._reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            vector_weight=0.7,
            bm25_weight=0.3,
            k=60,
        )

        # Should have 4 unique chunks: c1, c2, c3, c4
        chunk_ids = {r.chunk_id for r in fused}
        assert chunk_ids == {"c1", "c2", "c3", "c4"}

        # c1 and c2 appear in both lists → should rank higher
        # c2 is rank 0 in BM25 and rank 1 in vector → good combined score
        # c1 is rank 0 in vector and rank 2 in BM25 → good combined score
        top_two = {fused[0].chunk_id, fused[1].chunk_id}
        assert "c1" in top_two or "c2" in top_two

    def test_rrf_preserves_original_scores(self):
        """RRF results should carry both vector and BM25 original scores."""
        vector_results = [
            VectorSearchResult(id="v1", score=0.9, payload={"chunk_id": "c1", "content": "A"}),
        ]
        bm25_results = [
            BM25SearchResult(chunk_id="c1", score=7.0, payload={"chunk_id": "c1", "content": "A"}),
        ]

        fused = HybridRetriever._reciprocal_rank_fusion(
            vector_results, bm25_results, k=60
        )

        assert len(fused) == 1
        assert fused[0].vector_score == 0.9
        assert fused[0].bm25_score == 7.0
        assert fused[0].score > 0

    def test_rrf_empty_inputs(self):
        """Empty inputs should produce empty output."""
        fused = HybridRetriever._reciprocal_rank_fusion([], [])
        assert fused == []

    def test_rrf_single_source_vector_only(self):
        """When only vector results exist, all should still get RRF scores."""
        vector_results = [
            VectorSearchResult(id="v1", score=0.9, payload={"chunk_id": "c1", "content": "A"}),
            VectorSearchResult(id="v2", score=0.8, payload={"chunk_id": "c2", "content": "B"}),
        ]

        fused = HybridRetriever._reciprocal_rank_fusion(
            vector_results, [], vector_weight=0.7, bm25_weight=0.3, k=60
        )

        assert len(fused) == 2
        # Rank 0 should score higher than rank 1
        assert fused[0].score > fused[1].score

    def test_rrf_scoring_formula(self):
        """Verify the RRF formula is correctly applied."""
        vector_results = [
            VectorSearchResult(id="v1", score=0.9, payload={"chunk_id": "c1", "content": "A"}),
        ]
        bm25_results = [
            BM25SearchResult(chunk_id="c1", score=5.0, payload={"chunk_id": "c1", "content": "A"}),
        ]

        k = 60
        fused = HybridRetriever._reciprocal_rank_fusion(
            vector_results, bm25_results, vector_weight=0.7, bm25_weight=0.3, k=k
        )

        # c1 is rank 0 in both lists
        expected_score = 0.7 / (k + 0) + 0.3 / (k + 0)
        assert abs(fused[0].score - expected_score) < 1e-6


class TestHybridRetriever:
    """Integration-style tests for HybridRetriever."""

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
                id="v1", score=0.95,
                payload={"chunk_id": "c1", "document_id": "d1", "content": "ML content"},
            ),
        ]
        return store

    @pytest.mark.asyncio
    async def test_vector_only_fallback(self, mock_embedding_provider, mock_vector_store):
        """When BM25 index isn't built, should fall back to vector-only."""
        with patch("backend.retrieval.hybrid_retriever.get_settings") as mock_settings:
            settings = MagicMock()
            settings.retrieval_strategy = "hybrid"
            settings.bm25_enabled = True
            settings.bm25_top_k = 20
            settings.vector_weight = 0.7
            settings.bm25_weight = 0.3
            settings.rrf_k = 60
            mock_settings.return_value = settings

            retriever = HybridRetriever(mock_embedding_provider, mock_vector_store)
            results = await retriever.retrieve("test query", top_k=5)

            assert len(results) == 1
            assert results[0].chunk_id == "v1"
