"""
Google Gemini LLM provider with retry logic, timeouts, and system_instruction support.
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


class GeminiLLM(BaseLLM):
    """LLM provider using Google's Gemini API with retry and timeout."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.6-flash",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        super().__init__()
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        client = self._get_client()

        # Separate system instruction from conversation messages
        system_parts, conversation = self._split_system_messages(messages)

        # Convert remaining messages to Gemini Content objects
        gemini_contents = self._convert_to_contents(conversation)

        # Build config with native system instruction
        config = self._build_config(
            system_parts=system_parts,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=self._model,
                        contents=gemini_contents,
                        config=config,
                    ),
                    timeout=self._timeout,
                )

                result = response.text or ""
                usage = getattr(response, "usage_metadata", None)
                self._last_usage = {
                    "input_tokens": getattr(usage, "prompt_token_count", 0),
                    "output_tokens": getattr(usage, "candidates_token_count", 0),
                    "total_tokens": getattr(usage, "total_token_count", 0),
                } if usage else {}
                # Rough token estimate for tracking (Gemini doesn't always expose usage)
                prompt_text = " ".join(m.get("content", "") for m in messages)
                self._total_tokens_used += len(result) // 4 + len(prompt_text) // 4
                logger.info("Gemini response: model=%s, attempt=%d", self._model, attempt + 1)
                return result

            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"Gemini API call timed out after {self._timeout}s"
                )
                logger.warning(
                    "Gemini timeout on attempt %d/%d", attempt + 1, _MAX_RETRIES
                )
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                # Only retry on transient errors
                is_transient = any(
                    kw in err_str
                    for kw in ("rate limit", "quota", "503", "500", "overloaded", "resource")
                )
                if not is_transient:
                    raise  # Non-transient errors fail immediately
                logger.warning(
                    "Gemini transient error on attempt %d/%d: %s",
                    attempt + 1, _MAX_RETRIES, e,
                )

            # Wait before retry (skip on last attempt)
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_RETRY_DELAYS[attempt])

        raise last_error or RuntimeError("Gemini LLM failed after all retries")

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        client = self._get_client()

        system_parts, conversation = self._split_system_messages(messages)
        gemini_contents = self._convert_to_contents(conversation)

        config = self._build_config(
            system_parts=system_parts,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content_stream(
                        model=self._model,
                        contents=gemini_contents,
                        config=config,
                    ),
                    timeout=self._timeout,
                )

                async for chunk in response:
                    if chunk.text:
                        yield chunk.text

                return

            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"Gemini API stream timed out after {self._timeout}s"
                )
                logger.warning(
                    "Gemini stream timeout on attempt %d/%d", attempt + 1, _MAX_RETRIES
                )
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_transient = any(
                    kw in err_str
                    for kw in ("rate limit", "quota", "503", "500", "overloaded", "resource")
                )
                if not is_transient:
                    raise
                logger.warning(
                    "Gemini stream transient error on attempt %d/%d: %s",
                    attempt + 1, _MAX_RETRIES, e,
                )

            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_RETRY_DELAYS[attempt])

        raise last_error or RuntimeError("Gemini LLM stream failed after all retries")

    def _build_config(
        self,
        system_parts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        """Build GenerateContentConfig with native system_instruction and JSON support."""
        from google.genai import types

        config_kwargs = {
            "temperature": temperature or self._temperature,
            "max_output_tokens": max_tokens or self._max_tokens,
        }

        # Native system instruction — Gemini properly enforces these
        if system_parts:
            config_kwargs["system_instruction"] = "\n\n".join(system_parts)

        # JSON response format support — critical for router, agent planner, decomposer
        response_format = kwargs.get("response_format")
        if response_format and isinstance(response_format, dict):
            if response_format.get("type") == "json_object":
                config_kwargs["response_mime_type"] = "application/json"

        return types.GenerateContentConfig(**config_kwargs)

    @staticmethod
    def _split_system_messages(
        messages: List[Dict[str, str]],
    ) -> tuple[List[str], List[Dict[str, str]]]:
        """Separate system messages from user/assistant messages."""
        system_parts = []
        conversation = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg.get("content", ""))
            else:
                conversation.append(msg)
        return system_parts, conversation

    @staticmethod
    def _convert_to_contents(messages: List[Dict[str, str]]) -> list:
        """
        Convert OpenAI-style messages to Gemini Content objects.

        Uses structured Content/Part objects for proper multi-turn conversation
        handling, instead of flattening to a single string.
        """
        from google.genai import types

        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Gemini uses "user" and "model" roles (not "assistant")
            gemini_role = "model" if role == "assistant" else "user"

            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part(text=content)],
                )
            )

        return contents

    @property
    def model_name(self) -> str:
        return self._model
