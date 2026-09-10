"""Shared test fixtures and configuration."""

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Provide test settings without requiring .env file."""
    with patch("backend.config.settings._settings", None):
        with patch.dict("os.environ", {
            "APP_ENV": "test",
            "DEBUG": "true",
            "DATABASE_PROVIDER": "sqlite",
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5432",
            "POSTGRES_USER": "test_user",
            "POSTGRES_PASSWORD": "test_pass",
            "POSTGRES_DB": "test_db",
            "VECTOR_DB_PROVIDER": "chroma",
            "EMBEDDING_PROVIDER": "sentence_transformer",
            "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-4o-mini",
            "OPENAI_API_KEY": "test-key",
            "UPLOAD_DIR": "./test_uploads",
            "CHUNK_SIZE": "256",
            "CHUNK_OVERLAP": "25",
        }):
            from backend.config.settings import Settings
            settings = Settings()
            yield settings


@pytest.fixture
def sample_pdf_text():
    """Sample text that might come from a PDF."""
    return """
    Machine Learning is a subset of Artificial Intelligence that enables 
    systems to learn and improve from experience without being explicitly 
    programmed. It focuses on the development of computer programs that 
    can access data and use it to learn for themselves.

    Supervised Learning involves training a model on labeled data, where 
    the desired output is known. Common algorithms include Linear 
    Regression, Decision Trees, and Neural Networks.

    Unsupervised Learning works with unlabeled data to discover hidden 
    patterns. Clustering and dimensionality reduction are key techniques.

    Deep Learning is a subset of machine learning that uses neural networks 
    with multiple layers (deep neural networks) to learn hierarchical 
    representations of data. It has been particularly successful in 
    image recognition, natural language processing, and speech recognition.
    """


@pytest.fixture
def sample_chunks():
    """Pre-generated sample chunks for testing."""
    return [
        {
            "content": "Machine Learning is a subset of AI that enables systems to learn.",
            "page_number": 1,
            "section": "Introduction",
            "content_type": "text",
        },
        {
            "content": "Supervised Learning involves training a model on labeled data.",
            "page_number": 1,
            "section": "Supervised Learning",
            "content_type": "text",
        },
        {
            "content": "Deep Learning uses neural networks with multiple layers.",
            "page_number": 2,
            "section": "Deep Learning",
            "content_type": "text",
        },
    ]
