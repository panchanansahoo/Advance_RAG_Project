from typing import Optional

from backend.config import get_settings
from backend.processing.vlm.base import BaseVLM

_vlm_instance: Optional[BaseVLM] = None


def get_vlm() -> BaseVLM:
    """
    Factory function to get the configured VLM instance.
    """
    global _vlm_instance
    if _vlm_instance is None:
        settings = get_settings()
        if settings.vlm_provider.lower() == "openai":
            from backend.processing.vlm.openai_vlm import OpenAIVLM
            api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
            _vlm_instance = OpenAIVLM(model_name=settings.vlm_model, api_key=api_key)
        else:
            from backend.processing.vlm.gemini_vlm import GeminiVLM
            api_key = settings.google_api_key.get_secret_value() if settings.google_api_key else None
            _vlm_instance = GeminiVLM(model_name=settings.vlm_model, api_key=api_key)

    return _vlm_instance

__all__ = ["BaseVLM", "get_vlm"]
