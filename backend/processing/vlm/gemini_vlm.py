"""
Gemini Vision Language Model integration.
"""

from __future__ import annotations

import logging
from typing import Optional
from PIL import Image

import google.generativeai as genai
from backend.processing.vlm.base import BaseVLM

logger = logging.getLogger(__name__)


class GeminiVLM(BaseVLM):
    """VLM implementation using Google Gemini Vision."""

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        """
        Args:
            model_name: The Gemini model to use.
            api_key: Optional API key. If not provided, it assumes `genai.configure()` was called globally.
        """
        self.model_name = model_name
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.default_prompt = (
            "Describe this image in detail. If it's a chart or graph, extract the data and explain the trends. "
            "If it contains a diagram, explain the flow or structure."
        )

    async def analyze_image(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        final_prompt = prompt or self.default_prompt
        try:
            # Gemini Python SDK doesn't natively support async image generation yet,
            # so we run it synchronously. A real async wrapper would use ThreadPoolExecutor.
            response = self.model.generate_content([final_prompt, image])
            return response.text.strip()
        except Exception as e:
            logger.error("Gemini VLM analysis failed: %s", e, exc_info=True)
            return ""
