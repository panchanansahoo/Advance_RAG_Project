"""Tests for the Google Gen AI based Gemini adapters."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.embeddings.gemini_embeddings import GeminiEmbeddingProvider
from backend.generation.llm.gemini_llm import GeminiLLM


@pytest.mark.asyncio
async def test_gemini_llm_uses_new_client_and_reports_usage():
    response = SimpleNamespace(
        text="Grounded Gemini answer",
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=8,
            total_token_count=20,
        ),
    )
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)

    llm = GeminiLLM(api_key="test-key")
    llm._client = client

    result = await llm.generate([{"role": "user", "content": "Question"}])

    assert result == "Grounded Gemini answer"
    assert llm.last_usage == {
        "input_tokens": 12,
        "output_tokens": 8,
        "total_tokens": 20,
    }
    client.aio.models.generate_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_gemini_embeddings_normalize_sdk_values():
    response = SimpleNamespace(
        embeddings=[
            SimpleNamespace(values=[0.1, 0.2]),
            SimpleNamespace(values=[0.3, 0.4]),
        ]
    )
    client = MagicMock()
    client.aio.models.embed_content = AsyncMock(return_value=response)

    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        model_name="gemini-embedding-001",
    )
    provider._client = client

    result = await provider.embed(["first", "second"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert provider.dimension == 3072
    client.aio.models.embed_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_gemini_embedding_query_returns_first_vector():
    response = SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5, 0.6])])
    client = MagicMock()
    client.aio.models.embed_content = AsyncMock(return_value=response)

    provider = GeminiEmbeddingProvider(api_key="test-key", dimension=2)
    provider._client = client

    assert await provider.embed_query("query") == [0.5, 0.6]