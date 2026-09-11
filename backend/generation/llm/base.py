"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator


class BaseLLM(ABC):
    """
    LLM provider abstraction.
    Implementations wrap specific LLM APIs (OpenAI, Gemini, Ollama, etc.).
    """

    def __init__(self):
        self._total_tokens_used: int = 0
        self._last_usage: Dict[str, int] = {}

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature override.
            max_tokens: Max output tokens override.

        Returns:
            The generated text response.
        """
        ...
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the generated response from the LLM.
        Yields string chunks of the generated response.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the LLM model."""
        ...

    @property
    def total_tokens_used(self) -> int:
        """Cumulative token usage across all calls for this provider instance."""
        return self._total_tokens_used

    @property
    def last_usage(self) -> Dict[str, int]:
        """Token usage reported by the most recent non-streaming call."""
        return dict(self._last_usage)
