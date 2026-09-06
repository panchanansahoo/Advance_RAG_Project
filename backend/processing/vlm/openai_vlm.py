"""
OpenAI Vision Language Model integration.
"""

from __future__ import annotations

import logging
import base64
from io import BytesIO
from typing import Optional
from PIL import Image

from openai import AsyncOpenAI
from backend.processing.vlm.base import BaseVLM

logger = logging.getLogger(__name__)


class OpenAIVLM(BaseVLM):
    """VLM implementation using OpenAI GPT-4 Vision."""

    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Args:
            model_name: The OpenAI vision model to use.
            api_key: Optional API key. If not provided, it uses the OPENAI_API_KEY env var.
        """
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()
        self.default_prompt = (
            "Describe this image in detail. If it's a chart or graph, extract the data and explain the trends. "
            "If it contains a diagram, explain the flow or structure."
        )

    def _encode_image(self, image: Image.Image) -> str:
        buffered = BytesIO()
        # Convert to RGB to avoid issues with saving alpha channels as JPEG
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    async def analyze_image(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        final_prompt = prompt or self.default_prompt
        try:
            base64_image = self._encode_image(image)
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": final_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("OpenAI VLM analysis failed: %s", e, exc_info=True)
            return ""
