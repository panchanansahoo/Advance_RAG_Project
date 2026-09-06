"""Tests for query processing (rewriting, expansion)."""

import pytest
from backend.processing.query.query_processor import ProcessedQuery, QueryProcessor


class TestProcessedQuery:
    """Tests for the ProcessedQuery dataclass."""

    def test_effective_query_with_rewrite(self):
        pq = ProcessedQuery(original="what is ML?", rewritten="What is machine learning?")
        assert pq.effective_query == "What is machine learning?"

    def test_effective_query_without_rewrite(self):
        pq = ProcessedQuery(original="what is ML?")
        assert pq.effective_query == "what is ML?"

    def test_all_queries_single(self):
        pq = ProcessedQuery(original="test")
        assert pq.all_queries == ["test"]

    def test_all_queries_with_sub_queries(self):
        pq = ProcessedQuery(
            original="compare A and B",
            sub_queries=["What is A?", "What is B?"],
        )
        assert len(pq.all_queries) == 3


class TestQueryProcessor:
    """Tests for QueryProcessor without LLM dependency."""

    def test_processor_creation(self):
        processor = QueryProcessor()
        assert processor is not None

    @pytest.mark.asyncio
    async def test_process_no_rewrite(self):
        """When rewriting is disabled, original query passes through."""
        processor = QueryProcessor()
        result = await processor.process(
            "What is deep learning?",
            rewrite=False,
            expand=False,
            decompose=False,
        )
        assert result.original == "What is deep learning?"
        assert result.rewritten is None
        assert result.effective_query == "What is deep learning?"
