"""
OpenAI LLM provider with retry logic, timeouts, and token usage tracking.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, AsyncGenerator

from backend.generation.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# Retry configuration
_MAX_RETRIES = 3
_RETRY_DELAYS = [1.0, 2.0, 4.0]  # Exponential backoff
_DEFAULT_TIMEOUT = 60  # seconds


class OpenAILLM(BaseLLM):
    """LLM provider using OpenAI's chat completions API with retry and timeout."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        base_url: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        super().__init__()
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        client = self._get_client()

        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        temperature=temperature or self._temperature,
                        max_tokens=max_tokens or self._max_tokens,
                        **kwargs,
                    ),
                    timeout=self._timeout,
                )

                result = response.choices[0].message.content or ""
                self._last_usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else {}
                tokens_used = self._last_usage.get("total_tokens", 0)
                self._total_tokens_used += tokens_used
                logger.info(
                    "OpenAI response: model=%s, tokens=%d, attempt=%d",
                    self._model, tokens_used, attempt + 1,
                )
                return result

            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"OpenAI API call timed out after {self._timeout}s"
                )
                logger.warning(
                    "OpenAI timeout on attempt %d/%d", attempt + 1, _MAX_RETRIES
                )
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                # Only retry on transient errors (rate limits, server errors)
                is_transient = any(
                    kw in err_str
                    for kw in ("rate limit", "429", "503", "500", "overloaded", "server_error")
                )
                if not is_transient:
                    raise  # Non-transient errors fail immediately
                logger.warning(
                    "OpenAI transient error on attempt %d/%d: %s",
                    attempt + 1, _MAX_RETRIES, e,
                )

            # Wait before retry (skip on last attempt)
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_RETRY_DELAYS[attempt])

        raise last_error or RuntimeError("OpenAI LLM failed after all retries")

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        client = self._get_client()

        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        temperature=temperature or self._temperature,
                        max_tokens=max_tokens or self._max_tokens,
                        stream=True,
                        **kwargs,
                    ),
                    timeout=self._timeout,
                )

                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
                
                return

            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"OpenAI API stream timed out after {self._timeout}s"
                )
                logger.warning(
                    "OpenAI stream timeout on attempt %d/%d", attempt + 1, _MAX_RETRIES
                )
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_transient = any(
                    kw in err_str
                    for kw in ("rate limit", "429", "503", "500", "overloaded", "server_error")
                )
                if not is_transient:
                    raise
                logger.warning(
                    "OpenAI stream transient error on attempt %d/%d: %s",
                    attempt + 1, _MAX_RETRIES, e,
                )

            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_RETRY_DELAYS[attempt])

        raise last_error or RuntimeError("OpenAI LLM stream failed after all retries")

    @property
    def model_name(self) -> str:
        return self._model
