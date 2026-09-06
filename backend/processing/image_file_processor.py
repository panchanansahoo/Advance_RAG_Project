"""
Processor for standalone image files (PNG, JPG, JPEG).

Phase 3 component. Extracts text via OCR and describes image via VLM.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from PIL import Image

from backend.processing.base import BaseProcessor, ProcessedContent
from backend.processing.ocr_processor import get_ocr_processor
from backend.processing.vlm import get_vlm
from backend.config import get_settings
from backend.schemas.common import ContentType

logger = logging.getLogger(__name__)


class ImageFileProcessor(BaseProcessor):
    """
    Processes image files by applying OCR and VLM analysis.
    """

    def supported_extensions(self) -> List[str]:
        return [".png", ".jpg", ".jpeg"]

    async def process(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[ProcessedContent]:
        """
        Process an image file.
        """
        metadata = metadata or {}
        contents: List[ProcessedContent] = []
        settings = get_settings()

        try:
            image = Image.open(file_path)
        except Exception as e:
            logger.error("Failed to open image file: %s", e)
            raise ValueError(f"Invalid image file: {e}")

        text_content = ""
        visual_desc = ""

        # Step 1: OCR
        if settings.ocr_enabled:
            ocr = get_ocr_processor()
            if ocr.is_available():
                extracted_text = ocr.extract_text_from_image(image)
                if extracted_text:
                    text_content += f"[OCR Extracted Text]\n{extracted_text}\n\n"
            else:
                logger.warning("OCR is enabled but Tesseract is not available.")

        # Step 2: VLM Analysis
        vlm = get_vlm()
        try:
            description = await vlm.analyze_image(
                image,
                prompt="Describe this image in detail. Extract any data or text present."
            )
            if description:
                visual_desc = description
                text_content += f"[VLM Image Description]\n{description}\n"
        except Exception as e:
            logger.warning("VLM analysis failed for image: %s", e)

        if text_content.strip():
            contents.append(
                ProcessedContent(
                    text=text_content.strip(),
                    content_type=ContentType.TEXT,
                    page_number=1,
                    visual_description=visual_desc,
                    metadata={
                        **metadata,
                        "type": "image",
                        "format": image.format,
                        "source": "ocr_vlm",
                    }
                )
            )

        logger.info("Image processed: %s", file_path)
        return contents
