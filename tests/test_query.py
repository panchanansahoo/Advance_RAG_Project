"""Tests for the query/generation pipeline."""

import pytest
from backend.generation.prompt_templates import (
    RAG_SYSTEM_PROMPT,
    build_messages,
    format_context,
)
from backend.schemas.chunks import RetrievedChunk
from backend.schemas.common import ContentType
from uuid import uuid4


class TestPromptTemplates:
    """Tests for prompt template formatting."""

    def test_system_prompt_has_grounding_rules(self):
        assert "Evidence-Grounded" in RAG_SYSTEM_PROMPT
        assert "Citations Required" in RAG_SYSTEM_PROMPT
        assert "No Hallucination" in RAG_SYSTEM_PROMPT

    def test_format_context_with_chunks(self):
        chunks = [
            RetrievedChunk(
                chunk_id=uuid4(),
                document_id=uuid4(),
                content="Machine learning is a field of AI.",
                content_type=ContentType.TEXT,
                page_number=1,
                source_filename="ml.pdf",
                score=0.95,
            ),
            RetrievedChunk(
                chunk_id=uuid4(),
                document_id=uuid4(),
                content="Deep learning uses neural networks.",
                content_type=ContentType.TEXT,
                page_number=3,
                source_filename="dl.pdf",
                score=0.88,
            ),
        ]

        result = format_context(chunks, "What is ML?")
        assert "[1]" in result
        assert "[2]" in result
        assert "ml.pdf" in result
        assert "dl.pdf" in result
        assert "What is ML?" in result

    def test_format_context_empty_chunks(self):
        result = format_context([], "What is ML?")
        assert "What is ML?" in result

    def test_build_messages_structure(self):
        messages = build_messages("Test context prompt")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Test context prompt" in messages[1]["content"]

    def test_context_includes_source_info(self):
        chunks = [
            RetrievedChunk(
                chunk_id=uuid4(),
                document_id=uuid4(),
                content="Test content",
                content_type=ContentType.TEXT,
                page_number=5,
                section="Introduction",
                source_filename="test.pdf",
                score=0.9,
            ),
        ]
        result = format_context(chunks, "query")
        assert "Page: 5" in result
        assert "Section: Introduction" in result
        assert "test.pdf" in result
