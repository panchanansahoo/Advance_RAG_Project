"""Tests for Query Router."""

import pytest
from unittest.mock import AsyncMock, patch

from backend.routing.query_router import QueryRouter, RouteType


class TestQueryRouter:
    """Tests for the QueryRouter."""

    @patch("backend.routing.query_router.get_settings")
    @pytest.mark.asyncio
    async def test_routing_disabled(self, mock_settings):
        settings = mock_settings.return_value
        settings.query_routing_enabled = False

        router = QueryRouter()
        route = await router.route_query("What is the total revenue?", [".csv"])
        assert route == RouteType.RAG

    @patch("backend.routing.query_router.get_settings")
    @pytest.mark.asyncio
    async def test_no_structured_data(self, mock_settings):
        settings = mock_settings.return_value
        settings.query_routing_enabled = True

        router = QueryRouter()
        route = await router.route_query("What is the total revenue?", [".pdf", ".txt"])
        assert route == RouteType.RAG

    @patch("backend.routing.query_router.get_llm")
    @patch("backend.routing.query_router.get_settings")
    @pytest.mark.asyncio
    async def test_route_to_structured_data(self, mock_settings, mock_get_llm):
        settings = mock_settings.return_value
        settings.query_routing_enabled = True

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = '{"route": "structured_data", "reasoning": "Asking for numerical sum."}'
        mock_get_llm.return_value = mock_llm

        router = QueryRouter()
        route = await router.route_query("What is the total revenue?", [".csv"])
        
        assert route == RouteType.STRUCTURED_DATA
        mock_llm.generate.assert_called_once()

    @patch("backend.routing.query_router.get_llm")
    @patch("backend.routing.query_router.get_settings")
    @pytest.mark.asyncio
    async def test_route_to_rag(self, mock_settings, mock_get_llm):
        settings = mock_settings.return_value
        settings.query_routing_enabled = True

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = '{"route": "rag", "reasoning": "Asking for summary."}'
        mock_get_llm.return_value = mock_llm

        router = QueryRouter()
        route = await router.route_query("Summarize this document.", [".csv"])
        
        assert route == RouteType.RAG

    @patch("backend.routing.query_router.get_llm")
    @patch("backend.routing.query_router.get_settings")
    @pytest.mark.asyncio
    async def test_route_fallback_on_error(self, mock_settings, mock_get_llm):
        settings = mock_settings.return_value
        settings.query_routing_enabled = True

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = Exception("LLM Error")
        mock_get_llm.return_value = mock_llm

        router = QueryRouter()
        route = await router.route_query("What is the total revenue?", [".csv"])
        
        assert route == RouteType.RAG

    @patch("backend.routing.query_router.get_llm")
    @patch("backend.routing.query_router.get_settings")
    @pytest.mark.asyncio
    async def test_route_to_visual(self, mock_settings, mock_get_llm):
        settings = mock_settings.return_value
        settings.query_routing_enabled = True

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = '{"route": "visual", "reasoning": "Asking about chart."}'
        mock_get_llm.return_value = mock_llm

        router = QueryRouter()
        route = await router.route_query("What does the chart show?", [".png"])
        assert route == RouteType.VISUAL

    @patch("backend.routing.query_router.get_llm")
    @patch("backend.routing.query_router.get_settings")
    @pytest.mark.asyncio
    async def test_route_to_agentic(self, mock_settings, mock_get_llm):
        settings = mock_settings.return_value
        settings.query_routing_enabled = True

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = '{"route": "agentic", "reasoning": "Complex multi-hop reasoning."}'
        mock_get_llm.return_value = mock_llm

        router = QueryRouter()
        route = await router.route_query("Compare Report A and Report B step by step", [".pdf"])
        assert route == RouteType.AGENTIC

    def test_get_query_router_singleton(self):
        from backend.routing.query_router import get_query_router
        r1 = get_query_router()
        r2 = get_query_router()
        assert r1 is r2

