"""
Sentence-based chunking using NLTK sentence tokenizer.
Groups sentences until the target chunk size is reached.
"""

from __future__ import annotations

import logging
from typing import List

from backend.processing.chunking.base import BaseChunker, ChunkData
from backend.schemas.common import ContentType

logger = logging.getLogger(__name__)

# Lazy-load NLTK data
_nltk_ready = False


def _ensure_nltk():
    global _nltk_ready
    if not _nltk_ready:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        _nltk_ready = True


class SentenceChunker(BaseChunker):
    """
    Split text into chunks at sentence boundaries.
    Sentences are grouped until the target chunk size is reached.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 1):
        """
        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Number of sentences to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.sentence_overlap = chunk_overlap

    @property
    def strategy_name(self) -> str:
        return "sentence"

    def chunk(self, text: str, **kwargs) -> List[ChunkData]:
        if not text or not text.strip():
            return []

        _ensure_nltk()
        from nltk.tokenize import sent_tokenize

        page_number = kwargs.get("page_number")
        section = kwargs.get("section")
        content_type = kwargs.get("content_type", ContentType.TEXT)
        start_index = kwargs.get("start_index", 0)

        sentences = sent_tokenize(text.strip())
        if not sentences:
            return []

        chunks: List[ChunkData] = []
        current_sentences: List[str] = []
        current_length = 0
        idx = start_index

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence would exceed the target, flush current chunk
            if current_length + sentence_length > self.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                if chunk_text:
                    chunks.append(
                        ChunkData(
                            content=chunk_text,
                            chunk_index=idx,
                            content_type=content_type,
                            page_number=page_number,
                            section=section,
                            token_count=len(chunk_text.split()),
                        )
                    )
                    idx += 1

                # Keep overlap sentences
                if self.sentence_overlap > 0 and len(current_sentences) > self.sentence_overlap:
                    current_sentences = current_sentences[-self.sentence_overlap:]
                    current_length = sum(len(s) for s in current_sentences)
                else:
                    current_sentences = []
                    current_length = 0

            current_sentences.append(sentence)
            current_length += sentence_length

        # Flush remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(
                    ChunkData(
                        content=chunk_text,
                        chunk_index=idx,
                        content_type=content_type,
                        page_number=page_number,
                        section=section,
                        token_count=len(chunk_text.split()),
                    )
                )

        return chunks
