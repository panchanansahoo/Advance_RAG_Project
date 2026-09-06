"""Tests for the cross-encoder reranker."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.retrieval.reranker import CrossEncoderReranker, RerankerResult


class TestRerankerResult:
    """Tests for the RerankerResult dataclass."""

    def test_creation(self):
        r = RerankerResult(
            chunk_id="c1", score=0.95, original_score=0.8, payload={"content": "test"}
        )
        assert r.chunk_id == "c1"
        assert r.score == 0.95
        assert r.original_score == 0.8

    def test_repr(self):
        r = RerankerResult(chunk_id="c1", score=0.95, original_score=0.8, payload={})
        assert "c1" in repr(r)


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker with mocked model."""

    @pytest.mark.asyncio
    async def test_rerank_empty_results(self):
        reranker = CrossEncoderReranker()
        results = await reranker.rerank("test query", [], top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_sorts_by_score(self):
        """Test that reranking sorts by cross-encoder score."""
        reranker = CrossEncoderReranker()

        # Mock the model
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.3, 0.9, 0.1]  # Scores for 3 inputs
        reranker._model = mock_model

        results = [
            MagicMock(chunk_id="c1", score=0.95, payload={"content": "First chunk"}),
            MagicMock(chunk_id="c2", score=0.85, payload={"content": "Second chunk"}),
            MagicMock(chunk_id="c3", score=0.75, payload={"content": "Third chunk"}),
        ]

        reranked = await reranker.rerank("test query", results, top_k=3)

        assert len(reranked) == 3
        # c2 should be first (score 0.9)
        assert reranked[0].chunk_id == "c2"
        assert reranked[0].score == 0.9
        # c1 should be second (score 0.3)
        assert reranked[1].chunk_id == "c1"
        assert reranked[1].score == 0.3

    @pytest.mark.asyncio
    async def test_rerank_top_k_limiting(self):
        """Reranker should only return top_k results."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.9, 0.1, 0.7, 0.3]
        reranker._model = mock_model

        results = [
            MagicMock(chunk_id=f"c{i}", score=0.5, payload={"content": f"Chunk {i}"})
            for i in range(5)
        ]

        reranked = await reranker.rerank("test", results, top_k=2)
        assert len(reranked) == 2
