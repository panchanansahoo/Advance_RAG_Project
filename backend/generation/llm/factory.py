"""Factory for creating the configured LLM provider."""

from __future__ import annotations

import logging

from backend.config import get_settings
from backend.generation.llm.base import BaseLLM

logger = logging.getLogger(__name__)

_instance: BaseLLM | None = None


def get_llm() -> BaseLLM:
    """Return a cached LLM provider based on settings."""
    global _instance
    if _instance is not None:
        return _instance

    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "gemini":
        from backend.generation.llm.gemini_llm import GeminiLLM

        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini LLM")
        _instance = GeminiLLM(
            api_key=settings.google_api_key.get_secret_value(),
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    elif provider == "groq":
        from backend.generation.llm.groq_llm import GroqLLM

        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for Groq LLM")
        _instance = GroqLLM(
            api_key=settings.groq_api_key.get_secret_value(),
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    else:  # default: openai
        from backend.generation.llm.openai_llm import OpenAILLM

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI LLM")
        _instance = OpenAILLM(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    logger.info("LLM provider: %s (model=%s)", provider, _instance.model_name)
    return _instance
