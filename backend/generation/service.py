"""
Generation service — orchestrates retrieval → context building → LLM → response.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, AsyncGenerator
from uuid import UUID

from backend.config import get_settings
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

_ESTIMATED_PRICING_PER_MILLION = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-3.5-turbo": (0.50, 1.50),
}


def _estimate_usage(messages: list[dict], answer: str, model_name: str) -> dict:
    """Estimate per-request token usage and cost when provider usage is unavailable."""
    input_tokens = sum(len(message.get("content", "")) for message in messages) // 4
    output_tokens = len(answer) // 4
    input_price, output_price = _ESTIMATED_PRICING_PER_MILLION.get(model_name, (0.0, 0.0))
    estimated_cost = (
        input_tokens * input_price + output_tokens * output_price
    ) / 1_000_000
    return {
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": round(estimated_cost, 8),
        "usage_source": "character_estimate",
    }


def _usage_metadata(llm, messages: list[dict], answer: str) -> dict:
    """Use provider-reported usage when available, otherwise estimate it."""
    usage = getattr(llm, "last_usage", {})
    if not usage:
        return _estimate_usage(messages, answer, llm.model_name)

    input_price, output_price = _ESTIMATED_PRICING_PER_MILLION.get(
        llm.model_name, (0.0, 0.0)
    )
    estimated_cost = (
        usage.get("input_tokens", 0) * input_price
        + usage.get("output_tokens", 0) * output_price
    ) / 1_000_000
    return {
        "estimated_input_tokens": usage.get("input_tokens", 0),
        "estimated_output_tokens": usage.get("output_tokens", 0),
        "estimated_total_tokens": usage.get("total_tokens", 0),
        "estimated_cost_usd": round(estimated_cost, 8),
        "usage_source": "provider_reported",
    }


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
        owner_key: Optional[str] = None,
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
        retrieval_started = time.perf_counter()
        chunks = await self.retrieval_service.retrieve_chunks(
            query=request.query,
            top_k=request.top_k,
            document_ids=request.document_ids,
            owner_key=owner_key,
        )
        retrieval_time = time.perf_counter() - retrieval_started

        if not chunks:
            llm = get_llm()
            system_prompt = (
                "You are an Advanced RAG Assistant known for providing **thorough, in-depth, and comprehensive** answers. "
                "The user asked a question, but no matching content "
                "was found in their uploaded documents (or no documents were selected).\n\n"
                "Instructions:\n"
                "1. Begin with a clear notice indicating that no matching information was found in the uploaded documents.\n"
                "2. Provide a **thorough, detailed, expert-level** answer to the user's question using your general AI knowledge under a clear callout: '> 💡 **General Knowledge Answer**: While your uploaded documents do not cover this question, here is a detailed answer based on AI knowledge:'\n"
                "   - Include definitions, explanations, examples, comparisons, and practical insights as appropriate.\n"
                "   - Structure complex topics with sub-headings and bullet points.\n"
                "   - Aim for responses that would satisfy a knowledgeable professional.\n"
                "3. Provide a '### Recommended Plan & Next Steps' section offering actionable suggestions (such as what types of documents or data to upload for grounded answers, recommended query refinements, or follow-up steps).\n"
            )
            conv_history = ""
            if conversation_messages:
                conv_history = format_conversation_history(conversation_messages)

            user_content = f"User Question: {request.query}"
            if conv_history:
                user_content = conv_history + "\n\n" + user_content

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            generation_started = time.perf_counter()
            answer = await llm.generate(messages)
            generation_time = time.perf_counter() - generation_started
            usage_metadata = _usage_metadata(llm, messages, answer)

            return QueryResponse(
                answer=answer,
                citations=[],
                query=request.query,
                conversation_id=request.conversation_id,
                retrieval_metadata={
                    "chunks_retrieved": 0,
                    "source": "general_knowledge",
                    "embedding_provider": get_settings().embedding_provider,
                    "llm_model": llm.model_name,
                    "retrieval_time_seconds": round(retrieval_time, 3),
                    "generation_time_seconds": round(generation_time, 3),
                    "conversation_context_used": bool(conversation_messages),
                    **usage_metadata,
                },
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
        generation_started = time.perf_counter()
        answer = await llm.generate(messages)
        generation_time = time.perf_counter() - generation_started
        usage_metadata = _usage_metadata(llm, messages, answer)

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
                "embedding_provider": get_settings().embedding_provider,
                "llm_model": llm.model_name,
                "retrieval_time_seconds": round(retrieval_time, 3),
                "generation_time_seconds": round(generation_time, 3),
                "conversation_context_used": bool(conversation_messages),
                **usage_metadata,
            },
        )

    async def answer_stream(
        self,
        request: QueryRequest,
        conversation_messages: Optional[list] = None,
        owner_key: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        import json
        # 1. Retrieve relevant chunks
        chunks = await self.retrieval_service.retrieve_chunks(
            query=request.query,
            top_k=request.top_k,
            document_ids=request.document_ids,
            owner_key=owner_key,
        )

        if not chunks:
            llm = get_llm()
            system_prompt = (
                "You are an Advanced RAG Assistant known for providing **thorough, in-depth, and comprehensive** answers. "
                "The user asked a question, but no matching content "
                "was found in their uploaded documents (or no documents were selected).\n\n"
                "Instructions:\n"
                "1. Begin with a clear notice indicating that no matching information was found in the uploaded documents.\n"
                "2. Provide a **thorough, detailed, expert-level** answer to the user's question using your general AI knowledge under a clear callout: '> 💡 **General Knowledge Answer**: While your uploaded documents do not cover this question, here is a detailed answer based on AI knowledge:'\n"
                "   - Include definitions, explanations, examples, comparisons, and practical insights as appropriate.\n"
                "   - Structure complex topics with sub-headings and bullet points.\n"
                "   - Aim for responses that would satisfy a knowledgeable professional.\n"
                "3. Provide a '### Recommended Plan & Next Steps' section offering actionable suggestions (such as what types of documents or data to upload for grounded answers, recommended query refinements, or follow-up steps).\n"
            )
            conv_history = ""
            if conversation_messages:
                conv_history = format_conversation_history(conversation_messages)

            user_content = f"User Question: {request.query}"
            if conv_history:
                user_content = conv_history + "\n\n" + user_content

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            yield json.dumps({"citations": []}) + "\n"
            streamed_answer = []
            async for text_chunk in llm.generate_stream(messages):
                streamed_answer.append(text_chunk)
                yield json.dumps({"chunk": text_chunk}) + "\n"
            yield json.dumps({
                "metadata": {
                    "chunks_retrieved": 0,
                    "source": "general_knowledge",
                    "llm_model": llm.model_name,
                    **_estimate_usage(messages, "".join(streamed_answer), llm.model_name),
                }
            }) + "\n"
            return

        # 2. Build citations and yield them first
        citations = self._build_citations(chunks)
        yield json.dumps({
            "citations": [c.model_dump(mode='json') for c in citations]
        }) + "\n"

        # 3. Build the context prompt
        context_prompt = format_context(chunks, request.query)

        # 4. Build conversation history (if available)
        conv_history = ""
        if conversation_messages:
            conv_history = format_conversation_history(conversation_messages)

        # 5. Build messages and stream
        messages = build_messages(context_prompt, conversation_history=conv_history)
        llm = get_llm()
        streamed_answer = []
        
        async for text_chunk in llm.generate_stream(messages):
            streamed_answer.append(text_chunk)
            yield json.dumps({"chunk": text_chunk}) + "\n"

        yield json.dumps({
            "metadata": {
                "chunks_retrieved": len(chunks),
                "llm_model": llm.model_name,
                **_estimate_usage(messages, "".join(streamed_answer), llm.model_name),
            }
        }) + "\n"

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

