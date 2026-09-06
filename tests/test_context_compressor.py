"""Tests for the context compressor."""

import pytest
from backend.retrieval.context_compressor import ContextCompressor


class FakeChunk:
    """Simple mock chunk for testing."""
    def __init__(self, content: str, score: float = 0.9):
        self.content = content
        self.score = score
        self.payload = {"content": content}


class TestContextCompressor:
    """Tests for ContextCompressor."""

    def test_basic_compression(self):
        compressor = ContextCompressor(max_tokens=100)
        chunks = [
            FakeChunk("This is the first chunk with some content about machine learning."),
            FakeChunk("This is the second chunk about deep learning and neural networks."),
        ]
        result = compressor.compress(chunks)
        assert len(result) >= 1

    def test_deduplication(self):
        compressor = ContextCompressor(max_tokens=1000, similarity_threshold=0.75)
        chunks = [
            FakeChunk("Machine learning is a subset of artificial intelligence that enables systems to learn from data."),
            FakeChunk("Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve."),  # near-duplicate
            FakeChunk("Deep learning uses neural networks with many layers for pattern recognition."),
        ]
        result = compressor.compress(chunks)
        # Should remove the near-duplicate
        assert len(result) < len(chunks)

    def test_token_budget_enforcement(self):
        compressor = ContextCompressor(max_tokens=20)
        chunks = [
            FakeChunk("Word " * 10),  # 10 tokens
            FakeChunk("Word " * 10),  # 10 tokens
            FakeChunk("Word " * 10),  # 10 tokens
        ]
        result = compressor.compress(chunks)
        # Should cap at ~20 tokens (2 chunks)
        total_tokens = sum(len(c.content.split()) for c in result)
        assert total_tokens <= 25  # some flexibility

    def test_empty_input(self):
        compressor = ContextCompressor()
        assert compressor.compress([]) == []

    def test_single_chunk_passes_through(self):
        compressor = ContextCompressor(max_tokens=1000)
        chunks = [FakeChunk("Single chunk content.")]
        result = compressor.compress(chunks)
        assert len(result) == 1
        assert result[0].content == "Single chunk content."

    def test_completely_different_chunks_not_deduped(self):
        compressor = ContextCompressor(max_tokens=10000, similarity_threshold=0.85)
        chunks = [
            FakeChunk("Machine learning uses algorithms to learn from data."),
            FakeChunk("Quantum computing leverages quantum mechanical phenomena."),
            FakeChunk("Biology studies living organisms and their interactions."),
        ]
        result = compressor.compress(chunks)
        assert len(result) == 3  # All unique, all should remain
