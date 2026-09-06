"""
Base interface for Vision Language Models (VLM).

Phase 3 component for analyzing charts, diagrams, and complex images.
"""

from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image


class BaseVLM(ABC):
    """Abstract base class for Vision Language Models."""

    @abstractmethod
    async def analyze_image(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """
        Analyze an image and return a text description.

        Args:
            image: PIL Image to analyze.
            prompt: Optional specific prompt (e.g., "Extract the data from this chart").

        Returns:
            Text description/analysis of the image.
        """
        pass
