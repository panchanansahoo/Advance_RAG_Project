"""Tests for DOCX Processor."""

import pytest
from unittest.mock import MagicMock, patch
from backend.processing.docx_processor import DOCXProcessor


class TestDOCXProcessor:
    """Tests for DOCXProcessor."""

    def test_supported_extensions(self):
        processor = DOCXProcessor()
        assert ".docx" in processor.supported_extensions()

    @patch("backend.processing.docx_processor.docx")
    @pytest.mark.asyncio
    async def test_process_success(self, mock_docx):
        mock_doc = MagicMock()
        mock_para = MagicMock()
        mock_para.text = "This is a paragraph."
        mock_doc.paragraphs = [mock_para]
        
        mock_table = MagicMock()
        mock_row = MagicMock()
        mock_cell = MagicMock()
        mock_cell.text = "Cell data"
        mock_row.cells = [mock_cell, mock_cell]
        mock_table.rows = [mock_row]
        mock_doc.tables = [mock_table]
        
        mock_docx.Document.return_value = mock_doc
        
        processor = DOCXProcessor()
        contents = await processor.process("dummy.docx")
        
        assert len(contents) == 1
        assert "This is a paragraph." in contents[0].text
        assert "Cell data | Cell data" in contents[0].text
        assert contents[0].metadata["source"] == "python-docx"

    @patch("backend.processing.docx_processor.docx")
    @pytest.mark.asyncio
    async def test_process_failure(self, mock_docx):
        mock_docx.Document.side_effect = Exception("Corrupt file")
        
        processor = DOCXProcessor()
        with pytest.raises(ValueError, match="Invalid DOCX file"):
            await processor.process("dummy.docx")
