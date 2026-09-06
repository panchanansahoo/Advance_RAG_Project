"""Tests for Graph Extractor."""

import pytest
from unittest.mock import AsyncMock, patch

from backend.processing.graph_extractor import GraphExtractor, GraphExtractionResult, Node, Edge


class TestGraphExtractor:
    """Tests for the GraphExtractor."""

    @patch("backend.processing.graph_extractor.get_settings")
    @pytest.mark.asyncio
    async def test_extraction_disabled(self, mock_settings):
        settings = mock_settings.return_value
        settings.graph_extraction_enabled = False

        extractor = GraphExtractor()
        # Should return immediately and not call LLM or neo4j
        await extractor.extract_and_store("text", "doc_1")
        # Just testing it doesn't crash or error

    @patch("backend.processing.graph_extractor.get_llm")
    @patch("backend.processing.graph_extractor.get_settings")
    @pytest.mark.asyncio
    async def test_extract_graph_from_text(self, mock_settings, mock_get_llm):
        settings = mock_settings.return_value
        settings.graph_extraction_enabled = True

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = '''
        {
            "nodes": [{"id": "Tim Cook", "label": "Person"}],
            "edges": [{"source": "Tim Cook", "target": "Apple", "type": "WORKS_FOR"}]
        }
        '''
        mock_get_llm.return_value = mock_llm

        extractor = GraphExtractor()
        result = await extractor._extract_graph_from_text("Tim Cook works for Apple.")
        
        assert len(result.nodes) == 1
        assert result.nodes[0].id == "Tim Cook"
        assert result.nodes[0].label == "Person"
        assert len(result.edges) == 1
        assert result.edges[0].source == "Tim Cook"

    @patch("backend.processing.graph_extractor.get_llm")
    @patch("backend.processing.graph_extractor.get_settings")
    @pytest.mark.asyncio
    async def test_extract_graph_llm_failure(self, mock_settings, mock_get_llm):
        settings = mock_settings.return_value
        settings.graph_extraction_enabled = True

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = Exception("LLM Error")
        mock_get_llm.return_value = mock_llm

        extractor = GraphExtractor()
        result = await extractor._extract_graph_from_text("text")
        
        # Should gracefully return empty lists
        assert len(result.nodes) == 0
        assert len(result.edges) == 0

    @patch("backend.processing.graph_extractor.get_graph_db")
    @pytest.mark.asyncio
    async def test_store_in_neo4j(self, mock_get_graph_db):
        from unittest.mock import MagicMock
        mock_driver = MagicMock()
        mock_session = AsyncMock()
        mock_driver.session.return_value.__aenter__.return_value = mock_session
        mock_get_graph_db.return_value = mock_driver

        extractor = GraphExtractor()
        graph_data = GraphExtractionResult(
            nodes=[Node(id="A", label="Test")],
            edges=[Edge(source="A", target="B", type="RELS")]
        )
        
        await extractor._store_in_neo4j(graph_data, "doc_1")
        
        # 1 node + 1 edge = 2 cypher queries
        assert mock_session.run.call_count == 2
