"""Tests for Pandas Agent."""

import pytest
import pandas as pd
from unittest.mock import AsyncMock, patch

from backend.agent.pandas_agent import PandasAgent


class TestPandasAgent:
    """Tests for the PandasAgent."""

    @patch("backend.agent.pandas_agent.get_settings")
    @pytest.mark.asyncio
    async def test_run_no_files(self, mock_settings):
        agent = PandasAgent()
        result = await agent.run("What is the sum?", [])
        assert result == "No tabular files provided to analyze."

    @patch("backend.agent.pandas_agent.get_llm")
    @patch("backend.agent.pandas_agent.get_settings")
    @pytest.mark.asyncio
    async def test_run_success(self, mock_settings, mock_get_llm, tmp_path):
        settings = mock_settings.return_value
        settings.pandas_execution_timeout = 5

        # Create dummy CSV
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("A,B\n1,10\n2,20")

        mock_llm = AsyncMock()
        # First call: generate code
        # Second call: summarize
        mock_llm.generate.side_effect = [
            "```python\nresult = df_0['B'].sum()\n```",
            "The sum of column B is 30."
        ]
        mock_get_llm.return_value = mock_llm

        agent = PandasAgent()
        result = await agent.run("What is the sum of B?", [str(csv_file)])
        
        assert result == "The sum of column B is 30."

    @patch("backend.agent.pandas_agent.get_llm")
    @patch("backend.agent.pandas_agent.get_settings")
    @pytest.mark.asyncio
    async def test_security_import_block(self, mock_settings, mock_get_llm, tmp_path):
        settings = mock_settings.return_value
        settings.pandas_execution_timeout = 5

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("A,B\n1,10\n2,20")

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            "```python\nimport os\nresult = os.listdir('.')\n```",
            "Summary."
        ]
        mock_get_llm.return_value = mock_llm

        agent = PandasAgent()
        
        # It shouldn't crash, but _execute_code should return a security error string
        # which then gets summarized. Since we mock summarize, we can just check if _execute_code returns it
        # Actually let's just bypass run() and test _execute_code directly
        exec_result = await agent._execute_code("import os\nresult=1", {"df_0": {"path": str(csv_file)}})
        assert "Execution Error: Import of 'os' is not allowed for security reasons." in exec_result

    @patch("backend.agent.pandas_agent.asyncio.wait_for")
    @patch("backend.agent.pandas_agent.get_settings")
    @pytest.mark.asyncio
    async def test_timeout(self, mock_settings, mock_wait_for, tmp_path):
        import asyncio
        settings = mock_settings.return_value
        settings.pandas_execution_timeout = 1

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("A\n1")

        mock_wait_for.side_effect = asyncio.TimeoutError()

        agent = PandasAgent()
        
        exec_result = await agent._execute_code("result=1", {"df_0": {"path": str(csv_file)}})
        assert "Execution Timeout" in exec_result
