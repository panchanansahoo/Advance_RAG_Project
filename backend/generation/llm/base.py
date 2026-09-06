"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseLLM(ABC):
    """
    LLM provider abstraction.
    Implementations wrap specific LLM APIs (OpenAI, Gemini, Ollama, etc.).
    """

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

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the LLM model."""
        ...
