"""Tests for Image File Processor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image

from backend.processing.image_file_processor import ImageFileProcessor


class TestImageFileProcessor:
    """Tests for ImageFileProcessor."""

    def test_supported_extensions(self):
        processor = ImageFileProcessor()
        assert ".png" in processor.supported_extensions()
        assert ".jpg" in processor.supported_extensions()

    @patch("backend.processing.image_file_processor.Image")
    @patch("backend.processing.image_file_processor.get_ocr_processor")
    @patch("backend.processing.image_file_processor.get_vlm")
    @patch("backend.processing.image_file_processor.get_settings")
    @pytest.mark.asyncio
    async def test_process_success(self, mock_settings, mock_get_vlm, mock_get_ocr, mock_image_cls):
        settings = MagicMock()
        settings.ocr_enabled = True
        mock_settings.return_value = settings

        mock_ocr = MagicMock()
        mock_ocr.is_available.return_value = True
        mock_ocr.extract_text_from_image.return_value = "Test OCR"
        mock_get_ocr.return_value = mock_ocr

        mock_vlm = AsyncMock()
        mock_vlm.analyze_image.return_value = "Test VLM Description"
        mock_get_vlm.return_value = mock_vlm

        mock_image = MagicMock()
        mock_image.format = "PNG"
        mock_image_cls.open.return_value = mock_image

        processor = ImageFileProcessor()
        contents = await processor.process("dummy.png")
        
        assert len(contents) == 1
        assert "Test OCR" in contents[0].text
        assert "Test VLM Description" in contents[0].text
        assert contents[0].visual_description == "Test VLM Description"
        assert contents[0].metadata["type"] == "image"

    @patch("backend.processing.image_file_processor.Image")
    @pytest.mark.asyncio
    async def test_process_failure(self, mock_image_cls):
        mock_image_cls.open.side_effect = Exception("Invalid image")
        
        processor = ImageFileProcessor()
        with pytest.raises(ValueError, match="Invalid image file"):
            await processor.process("dummy.png")
