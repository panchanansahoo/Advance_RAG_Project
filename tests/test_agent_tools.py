"""Tests for Agent Tools."""

import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from backend.agent.tools import HybridSearchTool, GraphSearchTool, PandasQATool


class TestAgentTools:
    """Tests for the agent tools wrapper."""

    @patch("backend.agent.tools.RetrievalService")
    @pytest.mark.asyncio
    async def test_hybrid_search_tool_success(self, mock_retrieval_service):
        mock_service_instance = AsyncMock()
        mock_retrieval_service.return_value = mock_service_instance

        # Mock retrieved chunks
        chunk = type('Chunk', (), {
            'document_id': uuid4(),
            'source_filename': 'doc.txt',
            'chunk_id': uuid4(),
            'content': 'This is a test chunk.',
            'score': 0.9
        })()
        mock_service_instance.retrieve_chunks.return_value = [chunk]

        tool = HybridSearchTool()
        obs, citations, scores = await tool.execute("query")

        assert "[Doc: doc.txt] This is a test chunk." in obs
        assert "[Relevance: 0.900]" in obs
        assert len(citations) == 1
        assert citations[0]["document_name"] == "doc.txt"
        assert len(scores) == 1
        assert scores[0] == 0.9

    @patch("backend.agent.tools.RetrievalService")
    @pytest.mark.asyncio
    async def test_hybrid_search_tool_empty(self, mock_retrieval_service):
        mock_service_instance = AsyncMock()
        mock_retrieval_service.return_value = mock_service_instance
        mock_service_instance.retrieve_chunks.return_value = []

        tool = HybridSearchTool()
        obs, citations, scores = await tool.execute("query")

        assert "No relevant text documents found." in obs
        assert len(citations) == 0
        assert len(scores) == 0

    @patch("backend.agent.tools.GraphRetriever")
    @pytest.mark.asyncio
    async def test_graph_search_tool(self, mock_graph_retriever):
        mock_instance = AsyncMock()
        mock_graph_retriever.return_value = mock_instance
        mock_instance.search.return_value = [{"content": "A related to B", "score": 0.8}]

        tool = GraphSearchTool()
        obs, scores = await tool.execute("query")

        assert "A related to B" in obs
        assert "[Relevance: 0.800]" in obs
        assert len(scores) == 1
        assert scores[0] == 0.8

    @patch("backend.agent.tools.PandasAgent")
    @pytest.mark.asyncio
    async def test_pandas_qa_tool(self, mock_pandas_agent):
        mock_instance = AsyncMock()
        mock_pandas_agent.return_value = mock_instance
        mock_instance.run.return_value = "Sum is 10."

        tool = PandasQATool()
        obs = await tool.execute("query", ["a.csv"])

        assert "Sum is 10." in obs

    @pytest.mark.asyncio
    async def test_pandas_qa_tool_no_files(self):
        tool = PandasQATool()
        obs = await tool.execute("query", [])

        assert "No structured data files" in obs
