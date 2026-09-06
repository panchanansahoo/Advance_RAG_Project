"""Tests for Graph Retriever."""

import pytest
from unittest.mock import AsyncMock, patch

from backend.retrieval.graph_retriever import GraphRetriever


class TestGraphRetriever:
    """Tests for the GraphRetriever."""

    @patch("backend.retrieval.graph_retriever.get_graph_db")
    @pytest.mark.asyncio
    async def test_search_neo4j_not_configured(self, mock_get_graph_db):
        # If ValueError is raised, it means Neo4j isn't configured, should return []
        mock_get_graph_db.side_effect = ValueError("Neo4j connection failed")

        retriever = GraphRetriever()
        results = await retriever.search("query")
        assert results == []

    @patch("backend.retrieval.graph_retriever.get_llm")
    @patch("backend.retrieval.graph_retriever.get_graph_db")
    @pytest.mark.asyncio
    async def test_search_no_entities(self, mock_get_graph_db, mock_get_llm):
        mock_driver = AsyncMock()
        mock_get_graph_db.return_value = mock_driver

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "NONE"
        mock_get_llm.return_value = mock_llm

        retriever = GraphRetriever()
        results = await retriever.search("what is this")
        assert results == []

    @patch("backend.retrieval.graph_retriever.get_llm")
    @patch("backend.retrieval.graph_retriever.get_graph_db")
    @pytest.mark.asyncio
    async def test_search_success(self, mock_get_graph_db, mock_get_llm):
        from unittest.mock import MagicMock
        mock_driver = MagicMock()
        mock_session = AsyncMock()
        mock_driver.session.return_value.__aenter__.return_value = mock_session
        mock_get_graph_db.return_value = mock_driver

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "Apple, iPhone"
        mock_get_llm.return_value = mock_llm
        
        # Mocking the neo4j cursor
        async def mock_records():
            yield {"n_id": "Apple", "n_label": "Organization", "rel": "MAKES", "m_id": "iPhone", "m_label": "Product"}

        mock_result = AsyncMock()
        mock_result.__aiter__.side_effect = lambda: mock_records()
        mock_session.run.return_value = mock_result

        retriever = GraphRetriever()
        results = await retriever.search("Does Apple make iPhone?")
        
        assert len(results) == 1
        assert "Apple (Organization) MAKES iPhone (Product)" in results[0]["content"]
        assert results[0]["score"] == 0.8
        assert results[0]["source"] == "Knowledge Graph"
