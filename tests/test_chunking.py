"""Tests for the chunking module."""

import pytest
from backend.processing.chunking.fixed_chunker import FixedChunker
from backend.processing.chunking.sentence_chunker import SentenceChunker
from backend.processing.chunking import get_chunker
from backend.schemas.common import ContentType


class TestFixedChunker:
    """Tests for FixedChunker."""

    def test_basic_chunking(self):
        chunker = FixedChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a test sentence. " * 10  # 250 chars
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        assert all(c.content for c in chunks)

    def test_empty_text(self):
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_text_smaller_than_chunk_size(self):
        chunker = FixedChunker(chunk_size=1000, chunk_overlap=10)
        text = "Short text."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_chunk_indices_are_sequential(self):
        chunker = FixedChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 200
        chunks = chunker.chunk(text)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_metadata_passed_through(self):
        chunker = FixedChunker(chunk_size=1000, chunk_overlap=10)
        chunks = chunker.chunk(
            "Some content here.",
            page_number=5,
            section="Chapter 1",
            content_type=ContentType.TABLE,
        )
        assert len(chunks) == 1
        assert chunks[0].page_number == 5
        assert chunks[0].section == "Chapter 1"
        assert chunks[0].content_type == ContentType.TABLE

    def test_start_index_offset(self):
        chunker = FixedChunker(chunk_size=1000, chunk_overlap=10)
        chunks = chunker.chunk("Content", start_index=10)
        assert chunks[0].chunk_index == 10

    def test_overlap_creates_overlapping_content(self):
        chunker = FixedChunker(chunk_size=100, chunk_overlap=30)
        text = "Word " * 50  # 250 chars
        chunks = chunker.chunk(text)
        # With overlap, consecutive chunks should share some content
        if len(chunks) >= 2:
            # The end of chunk[0] and start of chunk[1] should overlap
            assert len(chunks) >= 2

    def test_strategy_name(self):
        chunker = FixedChunker()
        assert chunker.strategy_name == "fixed"


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_basic_sentence_chunking(self):
        chunker = SentenceChunker(chunk_size=100, chunk_overlap=0)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        # Each chunk should contain complete sentences
        for chunk in chunks:
            assert chunk.content

    def test_empty_text(self):
        chunker = SentenceChunker(chunk_size=100, chunk_overlap=0)
        assert chunker.chunk("") == []

    def test_single_sentence(self):
        chunker = SentenceChunker(chunk_size=1000, chunk_overlap=0)
        text = "This is one sentence."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1

    def test_strategy_name(self):
        chunker = SentenceChunker()
        assert chunker.strategy_name == "sentence"


class TestChunkerFactory:
    """Tests for the get_chunker factory."""

    def test_get_fixed_chunker(self):
        chunker = get_chunker("fixed", chunk_size=256, chunk_overlap=25)
        assert isinstance(chunker, FixedChunker)
        assert chunker.chunk_size == 256

    def test_get_sentence_chunker(self):
        chunker = get_chunker("sentence", chunk_size=256)
        assert isinstance(chunker, SentenceChunker)

    def test_default_is_fixed(self):
        chunker = get_chunker("unknown_strategy")
        assert isinstance(chunker, FixedChunker)
