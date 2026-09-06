"""
PDF processor using pdfplumber for text and table extraction.

Phase 1: handles text-based PDFs.
Phase 3: adds OCR (Tesseract) and VLM (Gemini/OpenAI) for scanned/visual PDFs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pdfplumber

from backend.processing.base import BaseProcessor, ProcessedContent
from backend.schemas.common import ContentType
from backend.config import get_settings
from backend.processing.ocr_processor import get_ocr_processor
from backend.processing.image_extractor import ImageExtractor
from backend.processing.vlm import get_vlm

logger = logging.getLogger(__name__)


class PDFProcessor(BaseProcessor):
    """Extract text, tables, and images from PDFs using pdfplumber, OCR, and VLM."""

    def __init__(self):
        self.settings = get_settings()
        self.image_extractor = ImageExtractor()

    def supported_extensions(self) -> List[str]:
        return [".pdf"]

    async def process(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[ProcessedContent]:
        contents: List[ProcessedContent] = []
        metadata = metadata or {}
        
        ocr = get_ocr_processor()
        vlm = get_vlm()

        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(
                    "Processing PDF: %s (%d pages)", file_path, total_pages
                )

                for page_num, page in enumerate(pdf.pages, start=1):
                    # ── Extract page text ───────────────────────
                    text = page.extract_text() or ""
                    
                    # Phase 3: OCR fallback if no text found (scanned PDF)
                    if not text.strip() and self.settings.ocr_enabled and ocr.is_available():
                        logger.info("No text found on page %d, trying OCR...", page_num)
                        page_image = page.to_image(resolution=300).original
                        text = ocr.extract_text_from_image(page_image)

                    if text.strip():
                        contents.append(
                            ProcessedContent(
                                text=text.strip(),
                                content_type=ContentType.TEXT,
                                page_number=page_num,
                                metadata={
                                    **metadata,
                                    "total_pages": total_pages,
                                    "source": "pdfplumber",
                                },
                            )
                        )

                    # ── Extract tables ──────────────────────────
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if not table:
                            continue

                        # Convert table to readable text
                        table_text = self._table_to_text(table)
                        if table_text.strip():
                            contents.append(
                                ProcessedContent(
                                    text=table_text,
                                    content_type=ContentType.TABLE,
                                    page_number=page_num,
                                    table_data={
                                        "headers": table[0] if table else [],
                                        "rows": table[1:] if len(table) > 1 else [],
                                        "table_index": table_idx,
                                    },
                                    metadata={
                                        **metadata,
                                        "total_pages": total_pages,
                                        "source": "pdfplumber",
                                    },
                                )
                            )

                    # ── Extract images (Phase 3) ────────────────
                    if self.settings.extract_images_from_pdf:
                        images = self.image_extractor.extract_images_from_page(page)
                        for img_idx, image in enumerate(images):
                            logger.info("Extracting image %d from page %d", img_idx, page_num)
                            
                            img_text = ""
                            visual_desc = ""
                            
                            # Analyze with VLM
                            try:
                                description = await vlm.analyze_image(
                                    image,
                                    prompt="Describe this image in detail. Extract any data or text present."
                                )
                                if description:
                                    visual_desc = description
                                    img_text += f"[VLM Image Description]\n{description}\n"
                            except Exception as e:
                                logger.warning("VLM analysis failed for image %d on page %d: %s", img_idx, page_num, e)
                                
                            if img_text.strip():
                                contents.append(
                                    ProcessedContent(
                                        text=img_text.strip(),
                                        content_type=ContentType.IMAGE,  # Task 4.3: use IMAGE, not TEXT
                                        page_number=page_num,
                                        visual_description=visual_desc,
                                        metadata={
                                            **metadata,
                                            "total_pages": total_pages,
                                            "source": "vlm",
                                            "image_index": img_idx,
                                        }
                                    )
                                )

        except Exception as e:
            logger.error("Failed to process PDF %s: %s", file_path, e)
            raise

        logger.info(
            "PDF processed: %s → %d content units", file_path, len(contents)
        )
        return contents

    @staticmethod
    def _table_to_text(table: List[List]) -> str:
        """Convert a table (list of rows) to a readable text representation."""
        if not table:
            return ""

        lines = []
        headers = table[0]
        if headers:
            lines.append(" | ".join(str(h or "") for h in headers))
            lines.append("-" * len(lines[0]))

        for row in table[1:]:
            if row:
                lines.append(" | ".join(str(cell or "") for cell in row))

        return "\n".join(lines)
