"""Tests for CSV Processor."""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

from backend.processing.csv_processor import CSVProcessor


class TestCSVProcessor:
    """Tests for CSVProcessor."""

    def test_supported_extensions(self):
        processor = CSVProcessor()
        assert ".csv" in processor.supported_extensions()
        assert ".xlsx" in processor.supported_extensions()

    @patch("backend.processing.csv_processor.pd")
    @pytest.mark.asyncio
    async def test_process_csv_success(self, mock_pd, tmp_path):
        mock_df = MagicMock()
        mock_df.to_markdown.return_value = "| Col1 | Col2 |\n|---|---|\n| A | B |"
        mock_df.__len__.return_value = 1
        mock_df.columns = ["Col1", "Col2"]
        mock_pd.read_csv.return_value = mock_df

        # Create a dummy CSV file
        dummy_file = tmp_path / "test.csv"
        dummy_file.write_text("Col1,Col2\nA,B")

        processor = CSVProcessor()
        contents = await processor.process(str(dummy_file))
        
        assert len(contents) == 1
        assert "| Col1 | Col2 |" in contents[0].text
        assert contents[0].metadata["file_type"] == "csv"

    @patch("backend.processing.csv_processor.pd")
    @pytest.mark.asyncio
    async def test_process_excel_success(self, mock_pd, tmp_path):
        mock_df = MagicMock()
        mock_df.to_markdown.return_value = "| Col1 | Col2 |\n|---|---|\n| A | B |"
        mock_df.__len__.return_value = 1
        mock_df.columns = ["Col1", "Col2"]
        mock_pd.read_excel.return_value = mock_df

        dummy_file = tmp_path / "test.xlsx"
        dummy_file.write_text("dummy")

        processor = CSVProcessor()
        contents = await processor.process(str(dummy_file))
        
        assert len(contents) == 1
        assert contents[0].metadata["file_type"] == "excel"

    @patch("backend.processing.csv_processor.pd")
    @pytest.mark.asyncio
    async def test_process_failure(self, mock_pd, tmp_path):
        mock_pd.read_excel.side_effect = Exception("Parse error")
        mock_pd.read_csv.side_effect = Exception("Parse error")
        
        dummy_file = tmp_path / "test.csv"
        dummy_file.write_text("dummy")

        processor = CSVProcessor()
        with pytest.raises(ValueError, match="Invalid or unsupported tabular data"):
            await processor.process(str(dummy_file))
