"""Tests for BM25 search module."""

import pytest
from backend.retrieval.bm25_search import BM25Index, _tokenize


class TestTokenizer:
    """Tests for the BM25 tokenizer."""

    def test_basic_tokenization(self):
        tokens = _tokenize("Machine learning is great")
        assert "machine" in tokens
        assert "learning" in tokens
        assert "great" in tokens

    def test_stopword_removal(self):
        tokens = _tokenize("the quick brown fox is very fast")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "very" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens

    def test_lowercasing(self):
        tokens = _tokenize("Machine LEARNING Deep")
        assert all(t.islower() for t in tokens)

    def test_punctuation_removal(self):
        tokens = _tokenize("hello, world! how's it going?")
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []
        assert _tokenize("   ") == []


class TestBM25Index:
    """Tests for the BM25 index."""

    @pytest.fixture
    def sample_chunks(self):
        return [
            {
                "chunk_id": "c1",
                "document_id": "d1",
                "content": "Machine learning is a subset of artificial intelligence that enables systems to learn.",
                "content_type": "text",
                "source_filename": "ml.pdf",
            },
            {
                "chunk_id": "c2",
                "document_id": "d1",
                "content": "Deep learning uses neural networks with multiple layers for complex pattern recognition.",
                "content_type": "text",
                "source_filename": "ml.pdf",
            },
            {
                "chunk_id": "c3",
                "document_id": "d2",
                "content": "Natural language processing deals with understanding and generating human language.",
                "content_type": "text",
                "source_filename": "nlp.pdf",
            },
            {
                "chunk_id": "c4",
                "document_id": "d2",
                "content": "Computer vision enables machines to interpret and understand visual information from images.",
                "content_type": "text",
                "source_filename": "cv.pdf",
            },
        ]

    def test_build_index(self, sample_chunks):
        index = BM25Index()
        index.build_index(sample_chunks)
        assert index.is_built
        assert index.size == 4

    def test_search_returns_results(self, sample_chunks):
        index = BM25Index()
        index.build_index(sample_chunks)
        results = index.search("machine learning neural networks", top_k=3)
        assert len(results) > 0
        # First results should be most relevant
        assert results[0].score > 0

    def test_search_ranking(self, sample_chunks):
        index = BM25Index()
        index.build_index(sample_chunks)
        results = index.search("deep learning neural networks", top_k=4)
        # "Deep learning uses neural networks..." should rank highest
        assert results[0].chunk_id == "c2"

    def test_search_with_document_filter(self, sample_chunks):
        index = BM25Index()
        index.build_index(sample_chunks)
        results = index.search("learning", top_k=10, document_ids=["d2"])
        # Should only return chunks from document d2
        for r in results:
            assert r.payload["document_id"] == "d2"

    def test_search_empty_index(self):
        index = BM25Index()
        results = index.search("test query")
        assert results == []

    def test_add_chunks(self, sample_chunks):
        index = BM25Index()
        index.build_index(sample_chunks[:2])
        assert index.size == 2

        index.add_chunks(sample_chunks[2:])
        assert index.size == 4

    def test_remove_document(self, sample_chunks):
        index = BM25Index()
        index.build_index(sample_chunks)
        assert index.size == 4

        index.remove_document("d1")
        assert index.size == 2

    def test_empty_build(self):
        index = BM25Index()
        index.build_index([])
        assert not index.is_built
