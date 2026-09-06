"""
Generation service — orchestrates retrieval → context building → LLM → response.
"""

from __future__ import annotations

import logging
from typing import List, Optional
from uuid import UUID

from backend.generation.llm import get_llm
from backend.generation.prompt_templates import (
    build_messages,
    format_context,
    format_conversation_history,
)
from backend.retrieval.service import RetrievalService, get_retrieval_service
from backend.schemas.chunks import RetrievedChunk
from backend.schemas.queries import Citation, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)


class GenerationService:
    """
    End-to-end query answering service:
    Query → Retrieve → Build Context → Generate → Format Response.
    """

    def __init__(self, retrieval_service: Optional[RetrievalService] = None):
        self.retrieval_service = retrieval_service or get_retrieval_service()

    async def answer(
        self,
        request: QueryRequest,
        conversation_messages: Optional[list] = None,
    ) -> QueryResponse:
        """
        Process a user query and generate a grounded answer.

        Args:
            request: The incoming QueryRequest with query, optional filters.
            conversation_messages: Optional list of prior Message objects for context.

        Returns:
            QueryResponse with answer, citations, and metadata.
        """
        # 1. Retrieve relevant chunks
        chunks = await self.retrieval_service.retrieve_chunks(
            query=request.query,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )

        if not chunks:
            return QueryResponse(
                answer="I couldn't find any relevant information in the uploaded documents to answer your question. Please upload relevant documents first or try rephrasing your question.",
                citations=[],
                query=request.query,
                conversation_id=request.conversation_id,
                retrieval_metadata={"chunks_retrieved": 0},
            )

        # 2. Build the context prompt
        context_prompt = format_context(chunks, request.query)

        # 3. Build conversation history (if available)
        conv_history = ""
        if conversation_messages:
            conv_history = format_conversation_history(conversation_messages)

        # 4. Build messages and generate
        messages = build_messages(context_prompt, conversation_history=conv_history)
        llm = get_llm()
        answer = await llm.generate(messages)

        # 5. Build citations from retrieved chunks
        citations = self._build_citations(chunks)

        # 6. Build response
        return QueryResponse(
            answer=answer,
            citations=citations,
            query=request.query,
            conversation_id=request.conversation_id,
            retrieval_metadata={
                "chunks_retrieved": len(chunks),
                "top_score": max(c.score for c in chunks) if chunks else 0,
                "min_score": min(c.score for c in chunks) if chunks else 0,
                "llm_model": llm.model_name,
                "conversation_context_used": bool(conversation_messages),
            },
        )

    @staticmethod
    def _build_citations(chunks: List[RetrievedChunk]) -> List[Citation]:
        """Convert retrieved chunks into citation objects."""
        citations = []
        for chunk in chunks:
            citations.append(
                Citation(
                    document_id=chunk.document_id,
                    document_name=chunk.source_filename or "Unknown",
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    content_snippet=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    relevance_score=chunk.score,
                )
            )
        return citations


# ── Singleton accessor (Task 4.1) ───────────────────────────
_generation_service: Optional[GenerationService] = None


def get_generation_service() -> GenerationService:
    """Return a cached GenerationService singleton."""
    global _generation_service
    if _generation_service is None:
        _generation_service = GenerationService()
    return _generation_service

