"""Tests for the OCR processor."""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from backend.processing.ocr_processor import OCRProcessor


class TestOCRProcessor:
    """Tests for OCRProcessor."""

    def test_processor_creation(self):
        processor = OCRProcessor()
        assert processor is not None

    @patch("backend.processing.ocr_processor.pytesseract")
    def test_extract_text_success(self, mock_tesseract):
        # Setup mock
        mock_tesseract.image_to_string.return_value = "Extracted OCR Text\n"
        
        processor = OCRProcessor()
        image = Image.new("RGB", (100, 100))
        
        text = processor.extract_text_from_image(image)
        
        assert text == "Extracted OCR Text"
        mock_tesseract.image_to_string.assert_called_once_with(image, lang="eng")

    @patch("backend.processing.ocr_processor.pytesseract")
    def test_extract_text_failure(self, mock_tesseract):
        # Setup mock to raise an exception
        mock_tesseract.image_to_string.side_effect = Exception("Tesseract not found")
        
        processor = OCRProcessor()
        image = Image.new("RGB", (100, 100))
        
        text = processor.extract_text_from_image(image)
        
        # Should catch exception and return empty string
        assert text == ""

    @patch("backend.processing.ocr_processor.pytesseract")
    def test_is_available_true(self, mock_tesseract):
        mock_tesseract.get_tesseract_version.return_value = "5.0.0"
        
        processor = OCRProcessor()
        assert processor.is_available() is True

    @patch("backend.processing.ocr_processor.pytesseract")
    def test_is_available_false(self, mock_tesseract):
        mock_tesseract.get_tesseract_version.side_effect = Exception("Not found")
        
        processor = OCRProcessor()
        assert processor.is_available() is False
