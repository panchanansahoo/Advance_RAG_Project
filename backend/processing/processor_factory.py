"""
Factory for selecting the correct document processor based on file extension.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from backend.processing.base import BaseProcessor
from backend.processing.pdf_processor import PDFProcessor
from backend.processing.text_processor import MarkdownProcessor, TextProcessor
from backend.processing.docx_processor import DOCXProcessor
from backend.processing.csv_processor import CSVProcessor
from backend.processing.image_file_processor import ImageFileProcessor
from backend.processing.html_processor import HTMLProcessor

logger = logging.getLogger(__name__)

# ── Registry ────────────────────────────────────────────────

_PROCESSORS: Dict[str, BaseProcessor] = {}


def _register_defaults():
    """Register all built-in processors."""
    global _PROCESSORS
    if _PROCESSORS:
        return

    for processor_cls in [
        PDFProcessor, TextProcessor, MarkdownProcessor,
        DOCXProcessor, CSVProcessor, ImageFileProcessor, HTMLProcessor
    ]:
        processor = processor_cls()
        for ext in processor.supported_extensions():
            _PROCESSORS[ext.lower()] = processor
            logger.debug("Registered processor for %s: %s", ext, processor_cls.__name__)


def get_processor(file_extension: str) -> Optional[BaseProcessor]:
    """
    Get the appropriate processor for a file extension.

    Args:
        file_extension: e.g. ".pdf", ".txt"

    Returns:
        A BaseProcessor instance, or None if unsupported.
    """
    _register_defaults()
    ext = file_extension.lower() if file_extension.startswith(".") else f".{file_extension.lower()}"
    return _PROCESSORS.get(ext)


def get_supported_extensions() -> list[str]:
    """Return all currently registered extensions."""
    _register_defaults()
    return list(_PROCESSORS.keys())
