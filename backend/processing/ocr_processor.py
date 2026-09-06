"""
OCR Processor for extracting text from images or scanned PDFs.

Phase 3 component using Tesseract OCR.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    Extracts text from images using Tesseract OCR.
    """

    def __init__(self):
        # Allow configuring tesseract path if needed
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pass

    def extract_text_from_image(self, image: Image.Image, lang: str = "eng") -> str:
        """
        Extract text from a single PIL Image.

        Args:
            image: PIL Image object.
            lang: Language string for tesseract.

        Returns:
            Extracted text string.
        """
        try:
            text = pytesseract.image_to_string(image, lang=lang)
            return text.strip()
        except Exception as e:
            logger.warning("OCR failed on image: %s", e)
            return ""

    def is_available(self) -> bool:
        """Check if Tesseract is installed and accessible."""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

# ── Singleton ───────────────────────────────────────────────

_ocr_processor: Optional[OCRProcessor] = None

def get_ocr_processor() -> OCRProcessor:
    """Return a cached OCRProcessor instance."""
    global _ocr_processor
    if _ocr_processor is None:
        _ocr_processor = OCRProcessor()
    return _ocr_processor
