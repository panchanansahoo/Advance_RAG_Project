"""
OpenAI LLM provider using the async OpenAI Python client.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from backend.generation.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """LLM provider using OpenAI's chat completions API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        base_url: Optional[str] = None,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        client = self._get_client()

        response = await client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature or self._temperature,
            max_tokens=max_tokens or self._max_tokens,
            **kwargs,
        )

        result = response.choices[0].message.content or ""
        logger.info(
            "OpenAI response: model=%s, tokens=%d",
            self._model,
            response.usage.total_tokens if response.usage else 0,
        )
        return result

    @property
    def model_name(self) -> str:
        return self._model
