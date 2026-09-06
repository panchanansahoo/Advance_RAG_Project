"""
Google Gemini LLM provider.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from backend.generation.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLM):
    """LLM provider using Google's Gemini API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self._model)
        return self._client

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        client = self._get_client()

        # Convert OpenAI-style messages to Gemini format
        gemini_messages = self._convert_messages(messages)

        response = await client.generate_content_async(
            gemini_messages,
            generation_config={
                "temperature": temperature or self._temperature,
                "max_output_tokens": max_tokens or self._max_tokens,
            },
        )

        result = response.text or ""
        logger.info("Gemini response: model=%s", self._model)
        return result

    @staticmethod
    def _convert_messages(messages: List[Dict[str, str]]) -> list:
        """
        Convert OpenAI-style messages to a single prompt string
        that Gemini can consume.
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System instructions: {content}\n\n")
            elif role == "user":
                parts.append(f"User: {content}\n\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n\n")
        return "".join(parts)

    @property
    def model_name(self) -> str:
        return self._model
