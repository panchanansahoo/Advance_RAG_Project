"""
Image extractor for extracting images and charts from PDF files.

Phase 3 component.
"""

from __future__ import annotations

import logging
from typing import List
from PIL import Image
import pdfplumber

logger = logging.getLogger(__name__)


class ImageExtractor:
    """
    Extracts images from pdfplumber page objects.
    """

    def extract_images_from_page(self, page: pdfplumber.page.Page) -> List[Image.Image]:
        """
        Extract all images from a given PDF page.
        """
        images = []
        try:
            for img_info in page.images:
                try:
                    # pdfplumber can extract the image bounding box as a crop
                    bbox = (img_info["x0"], img_info["top"], img_info["x1"], img_info["bottom"])
                    # Some bounding boxes might be invalid
                    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                        continue
                    
                    # Crop the page image
                    cropped = page.crop(bbox).to_image(resolution=200).original
                    images.append(cropped)
                except Exception as e:
                    logger.debug("Failed to extract single image from page: %s", e)
        except Exception as e:
            logger.warning("Failed to extract images from page %s: %s", page.page_number, e)
            
        return images
