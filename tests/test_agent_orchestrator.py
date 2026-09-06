"""Tests for Agent Orchestrator."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json

from backend.agent.orchestrator import AgentOrchestrator, _smart_truncate
from backend.schemas.agent import AgentAction, ToolCall


class TestAgentOrchestrator:
    """Tests for the AgentOrchestrator."""

    @patch("backend.agent.orchestrator.get_llm")
    @patch("backend.agent.orchestrator.get_settings")
    @pytest.mark.asyncio
    async def test_run_direct_final_answer(self, mock_settings, mock_get_llm):
        settings = mock_settings.return_value
        settings.agent_max_iterations = 5

        mock_llm = AsyncMock()
        # Call 1: decompose_query -> no decomposition needed
        # Call 2: plan action -> final_answer
        # Call 3: generate final answer
        mock_llm.generate.side_effect = [
            '{"needs_decomposition": false, "sub_questions": []}',
            '{"action": {"tool_name": "final_answer", "query": "Answer directly", "reasoning": "I know this."}}',
            "This is the final answer.",
        ]
        mock_get_llm.return_value = mock_llm

        orchestrator = AgentOrchestrator()
        state = await orchestrator.run("What is AI?")

        assert state.final_answer == "This is the final answer."
        assert len(state.steps) == 0

    @patch("backend.agent.orchestrator.get_llm")
    @patch("backend.agent.orchestrator.HybridSearchTool")
    @patch("backend.agent.orchestrator.get_settings")
    @pytest.mark.asyncio
    async def test_run_tool_then_answer(self, mock_settings, mock_hybrid_tool_class, mock_get_llm):
        settings = mock_settings.return_value
        settings.agent_max_iterations = 5

        mock_hybrid_tool = AsyncMock()
        mock_hybrid_tool.execute.return_value = ("Found text.", [{"chunk_id": "1"}], [0.85])
        mock_hybrid_tool_class.return_value = mock_hybrid_tool

        mock_llm = AsyncMock()
        # Call 1: decompose_query -> no decomposition
        # Call 2: plan action -> hybrid_search
        # Call 3: plan action -> final_answer
        # Call 4: generate final answer
        mock_llm.generate.side_effect = [
            '{"needs_decomposition": false, "sub_questions": []}',
            '{"action": {"tool_name": "hybrid_search", "query": "search AI", "reasoning": "need info"}}',
            '{"action": {"tool_name": "final_answer", "query": "summarize", "reasoning": "have info"}}',
            "This is the final answer.",
        ]
        mock_get_llm.return_value = mock_llm

        orchestrator = AgentOrchestrator()
        orchestrator.hybrid_tool = mock_hybrid_tool  # override for testing

        state = await orchestrator.run("What is AI?")

        assert state.final_answer == "This is the final answer."
        assert len(state.steps) == 1
        assert state.steps[0].tool == "hybrid_search"
        assert state.steps[0].observation == "Found text."
        assert state.steps[0].confidence > 0
        assert len(state.citations) == 1

    @patch("backend.agent.orchestrator.get_llm")
    @patch("backend.agent.orchestrator.get_settings")
    @pytest.mark.asyncio
    async def test_run_max_iterations(self, mock_settings, mock_get_llm):
        settings = mock_settings.return_value
        settings.agent_max_iterations = 2

        mock_llm = AsyncMock()
        # Call 1: decompose_query -> no decomposition
        # Calls 2+3: plan actions -> keep using graph_search
        # Final call: generate forced final answer
        mock_llm.generate.side_effect = [
            '{"needs_decomposition": false, "sub_questions": []}',
            '{"action": {"tool_name": "graph_search", "query": "AI", "reasoning": "info"}}',
            '{"action": {"tool_name": "graph_search", "query": "AI", "reasoning": "info"}}',
            "Forced final answer.",
        ]
        mock_get_llm.return_value = mock_llm

        orchestrator = AgentOrchestrator()
        # Mock the tool manually — now returns tuple (obs, scores)
        orchestrator.graph_tool = AsyncMock()
        orchestrator.graph_tool.execute.return_value = ("Graph obs", [0.5])

        state = await orchestrator.run("What is AI?")

        assert state.final_answer == "Forced final answer."
        assert len(state.steps) == 2  # It did 2 iterations then forced final answer

    @patch("backend.agent.orchestrator.get_llm")
    @patch("backend.agent.orchestrator.get_settings")
    @pytest.mark.asyncio
    async def test_query_decomposition(self, mock_settings, mock_get_llm):
        """Test that complex queries are decomposed into sub-questions."""
        settings = mock_settings.return_value
        settings.agent_max_iterations = 5

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            # Decomposition
            '{"needs_decomposition": true, "sub_questions": ["What is A?", "What is B?"]}',
            # Sub-Q1: plan -> final_answer
            '{"action": {"tool_name": "final_answer", "query": "A is X", "reasoning": "answered"}}',
            # Sub-Q2: plan -> final_answer
            '{"action": {"tool_name": "final_answer", "query": "B is Y", "reasoning": "answered"}}',
            # Final synthesis
            "A is X and B is Y.",
        ]
        mock_get_llm.return_value = mock_llm

        orchestrator = AgentOrchestrator()
        state = await orchestrator.run("Compare A and B")

        assert state.final_answer == "A is X and B is Y."
        assert len(state.sub_questions) == 2
        assert len(state.sub_answers) == 2


class TestSmartTruncate:
    """Tests for the intelligent truncation utility."""

    def test_short_text_not_truncated(self):
        text = "This is a short text."
        assert _smart_truncate(text, max_chars=100) == text

    def test_long_text_truncated_with_marker(self):
        text = "A" * 3000
        result = _smart_truncate(text, max_chars=2000)
        assert "[TRUNCATED" in result
        assert len(result) < 3000

    def test_truncation_at_sentence_boundary(self):
        text = "First sentence. Second sentence. " + "A" * 2000
        result = _smart_truncate(text, max_chars=50)
        assert "[TRUNCATED" in result
