"""
BM25 keyword search over chunked documents.

Maintains an in-memory BM25 index that is built from the PostgreSQL chunk
table. The index is rebuilt when new documents are ingested.

Phase 2 component: complements vector search for exact keyword matching.

Improvement (Task 2.4): Persistence via JSON serialization so the index
survives server restarts without requiring a full DB rebuild.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Default path for persisted BM25 index
_DEFAULT_INDEX_PATH = "./bm25_index.json"


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    # Remove very short tokens and stopwords
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
        "every", "all", "any", "few", "more", "most", "other", "some", "such",
        "than", "too", "very", "just", "about", "this", "that", "these",
        "those", "it", "its", "he", "she", "they", "them", "his", "her",
        "their", "we", "our", "you", "your", "i", "me", "my",
    }
    return [t for t in tokens if len(t) > 1 and t not in stopwords]


class BM25SearchResult:
    """A single result from BM25 search."""

    def __init__(self, chunk_id: str, score: float, payload: Dict[str, Any]):
        self.chunk_id = chunk_id
        self.score = score
        self.payload = payload

    def __repr__(self):
        return f"BM25SearchResult(chunk_id={self.chunk_id}, score={self.score:.4f})"


class BM25Index:
    """
    In-memory BM25 index over document chunks.

    The index stores chunk metadata alongside the BM25 corpus so that
    search results can be returned with full payload information.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: List[List[str]] = []
        self._chunk_payloads: List[Dict[str, Any]] = []
        self._is_built = False
        self._persist_path = persist_path or _DEFAULT_INDEX_PATH

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def size(self) -> int:
        return len(self._corpus)

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build the BM25 index from a list of chunk dicts.

        Each chunk dict must contain:
          - chunk_id: str
          - content: str
        And optionally: document_id, page_number, section, source_filename, content_type
        """
        if not chunks:
            logger.warning("No chunks provided to build BM25 index")
            return

        self._corpus = []
        self._chunk_payloads = []

        for chunk in chunks:
            content = chunk.get("content", "")
            tokens = _tokenize(content)
            if tokens:
                self._corpus.append(tokens)
                self._chunk_payloads.append(chunk)

        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)
            self._is_built = True
            self._save_to_disk()
            logger.info("BM25 index built with %d chunks", len(self._corpus))
        else:
            logger.warning("BM25 index is empty after tokenization")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add new chunks to the existing index.
        Rebuilds the BM25 index (BM25Okapi doesn't support incremental adds).
        """
        for chunk in chunks:
            content = chunk.get("content", "")
            tokens = _tokenize(content)
            if tokens:
                self._corpus.append(tokens)
                self._chunk_payloads.append(chunk)

        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)
            self._is_built = True
            self._save_to_disk()
            logger.info("BM25 index rebuilt with %d total chunks", len(self._corpus))

    def remove_document(self, document_id: str) -> None:
        """Remove all chunks for a document and rebuild the index."""
        new_corpus = []
        new_payloads = []

        for tokens, payload in zip(self._corpus, self._chunk_payloads):
            if payload.get("document_id") != document_id:
                new_corpus.append(tokens)
                new_payloads.append(payload)

        self._corpus = new_corpus
        self._chunk_payloads = new_payloads

        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)
        else:
            self._bm25 = None
            self._is_built = False

        self._save_to_disk()
        logger.info("Removed document %s from BM25 index (%d chunks remain)", document_id, len(self._corpus))

    def search(
        self,
        query: str,
        top_k: int = 20,
        document_ids: Optional[List[str]] = None,
    ) -> List[BM25SearchResult]:
        """
        Search the BM25 index.

        Args:
            query: The search query.
            top_k: Number of results to return.
            document_ids: Optional filter to specific documents.

        Returns:
            List of BM25SearchResult sorted by score descending.
        """
        if not self._is_built or self._bm25 is None:
            logger.warning("BM25 index not built, returning empty results")
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Get scores for all documents
        scores = self._bm25.get_scores(query_tokens)

        # Pair scores with payloads and filter
        scored_results: List[Tuple[float, Dict[str, Any]]] = []
        for i, (score, payload) in enumerate(zip(scores, self._chunk_payloads)):
            if score <= 0:
                continue

            # Apply document filter
            if document_ids and payload.get("document_id") not in document_ids:
                continue

            scored_results.append((score, payload))

        # Sort by score descending and take top-K
        scored_results.sort(key=lambda x: x[0], reverse=True)
        top_results = scored_results[:top_k]

        return [
            BM25SearchResult(
                chunk_id=payload.get("chunk_id", ""),
                score=score,
                payload=payload,
            )
            for score, payload in top_results
        ]


    # ── Persistence Methods (Task 2.4) ──────────────────────

    def _save_to_disk(self) -> None:
        """Persist the index payloads and corpus to a JSON file."""
        try:
            data = {
                "corpus": self._corpus,
                "payloads": self._chunk_payloads,
            }
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            logger.debug("BM25 index saved to %s (%d chunks)", self._persist_path, len(self._corpus))
        except Exception as e:
            logger.warning("Failed to persist BM25 index: %s", e)

    def load_from_disk(self) -> bool:
        """
        Load a previously persisted BM25 index from disk.

        Returns:
            True if the index was loaded successfully, False otherwise.
        """
        if not os.path.exists(self._persist_path):
            logger.debug("No persisted BM25 index found at %s", self._persist_path)
            return False

        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._corpus = data.get("corpus", [])
            self._chunk_payloads = data.get("payloads", [])

            if self._corpus:
                self._bm25 = BM25Okapi(self._corpus)
                self._is_built = True
                logger.info(
                    "BM25 index loaded from disk: %d chunks", len(self._corpus)
                )
                return True
            else:
                logger.debug("Persisted BM25 index was empty")
                return False

        except Exception as e:
            logger.warning("Failed to load persisted BM25 index: %s", e)
            return False


# ── Singleton BM25 Index ────────────────────────────────────

_bm25_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """Return the global BM25 index singleton."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index
