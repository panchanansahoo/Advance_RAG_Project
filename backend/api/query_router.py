"""
Query API router — the main QA endpoint.

Handles all 4 route types (PRD §11):
  RAG | Structured Data | Visual | Agentic

Task 1.3: Routes visual and agentic queries to their respective handlers.
Task 1.4: Passes full retrieved chunks (not truncated snippets) to verification.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.database.connection import get_db
from backend.database.models import Document
from backend.database.models import UsageEvent
from backend.database.repositories.conversation_repo import ConversationRepository
from backend.generation.service import GenerationService
from backend.schemas.queries import QueryRequest, QueryResponse
from backend.routing.query_router import QueryRouter, RouteType
from backend.agent.pandas_agent import PandasAgent
from backend.api.auth import enforce_question_access, get_user_key

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Query"])

# ── Rate Limiting ───────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


def _is_exempt_from_rate_limit(request: Request) -> bool:
    from backend.config import get_settings
    secret = request.headers.get("X-Internal-Secret")
    return bool(secret and secret == get_settings().auth_secret)


@router.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute", exempt_when=_is_exempt_from_rate_limit)
async def query(request_obj: QueryRequest, request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    """
    Ask a question and get a grounded answer with citations.
    Phase 4: Routes query to RAG, Pandas Agent, Visual, or Agentic based on context and intent.
    Phase 6: Agentic RAG with iterative tool use, decomposition, and self-correction.
    Phase 7: Verification and contradiction detection.
    §20: Conversation memory — loads and saves messages.
    """
    try:
        await enforce_question_access(request, response)
        owner_key = await get_user_key(request, response)
        from backend.api.usage import enforce_usage_limits
        await enforce_usage_limits(db, owner_key)
        query_start = time.perf_counter()
        # 0. Load conversation history (if conversation_id provided)
        conversation_messages = []
        conv_repo = ConversationRepository(db)

        if request_obj.conversation_id:
            try:
                conversation_messages = await conv_repo.get_recent_messages(
                    request_obj.conversation_id, limit=10, owner_key=owner_key
                )
            except Exception as e:
                logger.warning("Failed to load conversation history: %s", e)
        else:
            # Auto-create a conversation if none provided
            try:
                conv = await conv_repo.create_conversation(
                    title=request_obj.query[:80], owner_key=owner_key
                )
                request_obj = request_obj.model_copy(
                    update={"conversation_id": conv.id}
                )
            except Exception as e:
                logger.warning("Failed to auto-create conversation: %s", e)

        # 1. Fetch document extensions and paths if document_ids are provided
        available_extensions = []
        file_paths = []
        image_file_paths = []

        if request_obj.document_ids:
            result = await db.execute(
                select(Document).where(
                    Document.id.in_(request_obj.document_ids),
                    Document.owner_key == owner_key,
                )
            )
            docs = result.scalars().all()
            for doc in docs:
                ext = "." + doc.filename.split(".")[-1].lower() if "." in doc.filename else ""
                available_extensions.append(ext)
                if ext in [".csv", ".xlsx", ".xls"]:
                    file_paths.append(doc.file_path)
                if ext in [".png", ".jpg", ".jpeg"]:
                    image_file_paths.append(doc.file_path)

        # 2. Route the query (query-adaptive routing with overrides)
        from backend.config import get_settings
        settings = get_settings()

        route: Optional[RouteType] = None

        # Priority 1: Per-request explicit override
        if request_obj.force_route:
            try:
                route = RouteType(request_obj.force_route.lower())
                logger.info("Per-request forced route: %s", route)
            except ValueError:
                logger.warning(
                    "Unknown forced route '%s', falling back to adaptive routing",
                    request_obj.force_route,
                )
        elif request_obj.use_agent is True:
            route = RouteType.AGENTIC
            logger.info("Per-request forced agentic RAG via use_agent=True")
        elif request_obj.use_agent is False:
            route = RouteType.RAG
            logger.info("Per-request forced standard RAG via use_agent=False")

        # Priority 2: Global force override
        if route is None and getattr(settings, "agentic_rag_force", False):
            route = RouteType.AGENTIC
            logger.info("Agentic RAG globally forced via settings.agentic_rag_force")

        # Priority 3: Query-adaptive routing (default behavior)
        if route is None:
            from backend.routing.query_router import get_query_router
            router_svc = get_query_router()
            route = await router_svc.route_query(request_obj.query, available_extensions)
            # If router chose agentic, but agentic_rag_enabled is False, downgrade to RAG
            if route == RouteType.AGENTIC and not settings.agentic_rag_enabled:
                logger.info(
                    "Agentic RAG disabled in settings — downgrading adaptive agentic route to standard RAG"
                )
                route = RouteType.RAG

        # 3. Execute based on route
        response: QueryResponse

        if route == RouteType.AGENTIC:
            response = await _handle_agentic(
                request_obj, file_paths, settings, db, conversation_messages, owner_key
            )

        elif route == RouteType.STRUCTURED_DATA and file_paths:
            response = await _handle_structured_data(request_obj, file_paths)

        elif route == RouteType.VISUAL and image_file_paths:
            response = await _handle_visual(request_obj, image_file_paths, conversation_messages)

        else:
            # Default: standard RAG pipeline
            response = await _handle_rag(
                request_obj, conversation_messages, settings, owner_key
            )

        # 4. Clean answer to ensure no programming/traceback errors are presented
        from backend.utils.error_sanitizer import clean_response_answer, sanitize_error_message
        response.answer = clean_response_answer(response.answer)

        # 5. Record timing
        query_elapsed = time.perf_counter() - query_start
        response.retrieval_metadata["response_time_seconds"] = round(query_elapsed, 3)
        usage = response.retrieval_metadata
        try:
            db.add(UsageEvent(
                owner_key=owner_key,
                provider=settings.llm_provider,
                model=usage.get("llm_model", settings.llm_model),
                route=usage.get("route"),
                input_tokens=usage.get("estimated_input_tokens", 0),
                output_tokens=usage.get("estimated_output_tokens", 0),
                total_tokens=usage.get("estimated_total_tokens", 0),
                estimated_cost_usd=usage.get("estimated_cost_usd", 0.0),
                usage_source=usage.get("usage_source"),
            ))
            await db.commit()
        except Exception as usage_error:
            await db.rollback()
            logger.warning("Failed to record usage event: %s", usage_error)
        logger.info(
            "Query completed in %.2fs | route=%s | query='%s'",
            query_elapsed,
            response.retrieval_metadata.get("route", "unknown"),
            request_obj.query[:80],
        )

        # 6. Save conversation messages (§20)
        if request_obj.conversation_id:
            try:
                # Save user message
                await conv_repo.add_message(
                    conversation_id=request_obj.conversation_id,
                    role="user",
                    content=request_obj.query,
                    owner_key=owner_key,
                )
                # Save assistant response
                citation_data = [
                    {
                        "document_name": c.document_name,
                        "content_snippet": c.content_snippet,
                        "page_number": c.page_number,
                    }
                    for c in response.citations
                ]
                await conv_repo.add_message(
                    conversation_id=request_obj.conversation_id,
                    role="assistant",
                    content=response.answer,
                    citations=citation_data,
                    metadata=response.retrieval_metadata,
                    owner_key=owner_key,
                )
            except Exception as e:
                logger.warning("Failed to save conversation messages: %s", e)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Query failed: %s", e, exc_info=True)
        err_str = str(e).lower()
        if type(e).__name__ == "ResourceExhausted" or any(kw in err_str for kw in ("429", "quota", "rate limit")):
            raise HTTPException(
                status_code=429,
                detail="The AI assistant is experiencing high demand or has exceeded its quota. Please try again later.",
            )
        
        from backend.utils.error_sanitizer import sanitize_error_message
        friendly_detail = sanitize_error_message(
            e,
            default_fallback="We encountered an unexpected issue while researching your question. Please try asking again or rephrasing your prompt.",
        )
        raise HTTPException(
            status_code=500,
            detail=friendly_detail,
        )

@router.post("/query_stream")
@limiter.limit("10/minute")
async def query_stream(request_obj: QueryRequest, request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    """
    Stream standard RAG answers, or execute the full routed pipeline when a
    route is explicitly selected by the client.
    """
    try:
        await enforce_question_access(request, response)
        owner_key = await get_user_key(request, response)
        from backend.api.usage import enforce_usage_limits
        await enforce_usage_limits(db, owner_key)
        conversation_messages = []
        conv_repo = ConversationRepository(db)

        if request_obj.conversation_id:
            try:
                conversation_messages = await conv_repo.get_recent_messages(
                    request_obj.conversation_id, limit=10, owner_key=owner_key
                )
            except Exception as e:
                logger.warning("Failed to load conversation history: %s", e)
        else:
            try:
                conv = await conv_repo.create_conversation(
                    title=request_obj.query[:80], owner_key=owner_key
                )
                request_obj = request_obj.model_copy(
                    update={"conversation_id": conv.id}
                )
            except Exception as e:
                logger.warning("Failed to auto-create conversation: %s", e)

        from backend.generation.service import get_generation_service
        service = get_generation_service()

        async def stream_generator():
            try:
                import json

                streamed_answer = []
                streamed_citations = []
                streamed_metadata = {}
                if request_obj.conversation_id:
                    yield json.dumps({"conversation_id": str(request_obj.conversation_id)}) + "\n"

                if request_obj.force_route or request_obj.use_agent is not None:
                    routed_response = await query(request_obj, request, response, db)
                    yield json.dumps({
                        "citations": [citation.model_dump(mode="json") for citation in routed_response.citations],
                        "chunk": routed_response.answer,
                    }) + "\n"
                    return

                async for chunk in service.answer_stream(
                    request_obj,
                    conversation_messages=conversation_messages,
                    owner_key=owner_key,
                ):
                    yield chunk
                    try:
                        event = json.loads(chunk)
                        if event.get("chunk"):
                            streamed_answer.append(event["chunk"])
                        if event.get("citations"):
                            streamed_citations = event["citations"]
                        if event.get("metadata"):
                            streamed_metadata = event["metadata"]
                    except (TypeError, json.JSONDecodeError):
                        continue

                try:
                    from backend.config import get_settings
                    usage = streamed_metadata
                    settings = get_settings()
                    db.add(UsageEvent(
                        owner_key=owner_key,
                        provider=settings.llm_provider,
                        model=usage.get("llm_model", settings.llm_model),
                        route="rag",
                        input_tokens=usage.get("estimated_input_tokens", 0),
                        output_tokens=usage.get("estimated_output_tokens", 0),
                        total_tokens=usage.get("estimated_total_tokens", 0),
                        estimated_cost_usd=usage.get("estimated_cost_usd", 0.0),
                        usage_source=usage.get("usage_source"),
                    ))
                    await db.commit()
                except Exception as usage_error:
                    await db.rollback()
                    logger.warning("Failed to record streaming usage event: %s", usage_error)

                if request_obj.conversation_id:
                    await conv_repo.add_message(
                        conversation_id=request_obj.conversation_id,
                        role="user",
                        content=request_obj.query,
                        owner_key=owner_key,
                    )
                    await conv_repo.add_message(
                        conversation_id=request_obj.conversation_id,
                        role="assistant",
                        content="".join(streamed_answer),
                        citations=streamed_citations,
                        metadata=streamed_metadata,
                        owner_key=owner_key,
                    )
            except Exception as e:
                err_str = str(e).lower()
                if type(e).__name__ == "ResourceExhausted" or any(kw in err_str for kw in ("429", "quota", "rate limit")):
                    yield json.dumps({"error": "The AI assistant is experiencing high demand or has exceeded its quota. Please try again later."}) + "\n"
                    return
                
                from backend.utils.error_sanitizer import sanitize_error_message
                friendly_detail = sanitize_error_message(e, default_fallback="We encountered an unexpected issue while researching your question.")
                yield json.dumps({"error": friendly_detail}) + "\n"

        stream_response = StreamingResponse(stream_generator(), media_type="application/x-ndjson")
        for header in response.raw_headers:
            if header[0].lower() == b"set-cookie":
                stream_response.raw_headers.append(header)
        return stream_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Query stream failed: %s", e, exc_info=True)
        err_str = str(e).lower()
        if type(e).__name__ == "ResourceExhausted" or any(kw in err_str for kw in ("429", "quota", "rate limit")):
            raise HTTPException(
                status_code=429,
                detail="The AI assistant is experiencing high demand or has exceeded its quota. Please try again later.",
            )
        from backend.utils.error_sanitizer import sanitize_error_message
        friendly_detail = sanitize_error_message(e, default_fallback="We encountered an unexpected issue.")
        raise HTTPException(status_code=500, detail=friendly_detail)

# ── Route Handlers ──────────────────────────────────────────


async def _handle_agentic(
    request: QueryRequest,
    file_paths: list,
    settings,
    db: AsyncSession,
    conversation_messages: list = None,
    owner_key: str = None,
) -> QueryResponse:
    """Handle agentic RAG with decomposition and self-correction (Phase 6)."""
    logger.info("Executing via Agentic RAG Orchestrator")
    from backend.agent.orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator()
    state = await orchestrator.run(
        query=request.query,
        document_ids=request.document_ids,
        structured_file_paths=file_paths,
        conversation_messages=conversation_messages,
        owner_key=owner_key,
    )

    # Deduplicate citations
    unique_citations = {c["chunk_id"]: c for c in state.citations}.values()

    from backend.schemas.queries import Citation

    final_citations = []
    for c in unique_citations:
        final_citations.append(Citation(**c))

    final_answer = state.final_answer or "No answer generated."

    # Phase 7: Reliability & Verification (with full evidence, not truncated snippets)
    if settings.verification_enabled:
        from backend.verification.service import get_verification_service

        verifier = get_verification_service()
        # Pass full evidence from agent steps for proper fact-checking (Task 1.4)
        full_evidence_texts = [step.observation for step in state.steps]
        verification_result = await verifier.verify(
            request.query,
            final_answer,
            final_citations,
            full_evidence_texts=full_evidence_texts,
        )
        final_answer = verification_result.revised_answer or final_answer

    return QueryResponse(
        answer=final_answer,
        query=request.query,
        citations=final_citations,
        conversation_id=request.conversation_id,
        retrieval_metadata={
            "route": "agentic",
            "steps": len(state.steps),
            "sub_questions": len(state.sub_questions),
            "overall_confidence": state.overall_confidence,
        },
    )


async def _handle_structured_data(
    request: QueryRequest, file_paths: list
) -> QueryResponse:
    """Handle structured data queries via Pandas Agent (Phase 4)."""
    logger.info("Executing via Pandas Agent for %d files", len(file_paths))
    agent = PandasAgent()
    answer = await agent.run(request.query, file_paths)

    return QueryResponse(
        answer=answer,
        query=request.query,
        conversation_id=request.conversation_id,
        retrieval_metadata={
            "route": "structured_data",
            "files_analyzed": len(file_paths),
        },
    )


async def _handle_visual(
    request: QueryRequest,
    image_file_paths: list,
    conversation_messages: list,
) -> QueryResponse:
    """Handle visual queries using VLM (Phase 3 — new route)."""
    logger.info("Executing via Visual/VLM pipeline for %d images", len(image_file_paths))

    from backend.processing.vlm import get_vlm
    from PIL import Image

    vlm = get_vlm()
    results = []

    for img_path in image_file_paths[:3]:  # Cap at 3 images to control cost
        try:
            image = Image.open(img_path)
            description = await vlm.analyze_image(
                image,
                prompt=f"Answer this question about the image: {request.query}",
            )
            if description:
                import os
                results.append(f"[Image: {os.path.basename(img_path)}]\n{description}")
        except Exception as e:
            logger.warning("VLM analysis failed for %s: %s", img_path, e)

    if results:
        # Combine VLM results with standard RAG if available
        vlm_context = "\n\n".join(results)
        from backend.generation.llm import get_llm

        llm = get_llm()
        answer = await llm.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Use the visual analysis results below "
                        "to answer the user's question. If the visual analysis is insufficient, "
                        "say so explicitly."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {request.query}\n\n"
                        f"Visual Analysis Results:\n{vlm_context}"
                    ),
                },
            ],
            temperature=0.2,
        )
    else:
        answer = (
            "I couldn't analyze the visual content. "
            "Please ensure image files are uploaded and accessible."
        )

    return QueryResponse(
        answer=answer,
        query=request.query,
        conversation_id=request.conversation_id,
        retrieval_metadata={
            "route": "visual",
            "images_analyzed": len(results),
        },
    )


async def _handle_rag(
    request: QueryRequest,
    conversation_messages: list,
    settings,
    owner_key: str = None,
) -> QueryResponse:
    """Handle standard RAG pipeline (Phase 1-2)."""
    logger.info("Executing via standard RAG pipeline")
    from backend.generation.service import get_generation_service
    from backend.verification.service import get_verification_service

    service = get_generation_service()
    response = await service.answer(
        request,
        conversation_messages=conversation_messages,
        owner_key=owner_key,
    )
    response.retrieval_metadata["route"] = "rag"

    # Phase 7: Reliability & Verification
    if settings.verification_enabled:
        verification_started = time.perf_counter()
        verifier = get_verification_service()
        verification_result = await verifier.verify(
            request.query, response.answer, response.citations
        )
        response.answer = verification_result.revised_answer or response.answer
        response.retrieval_metadata["verification_time_seconds"] = round(
            time.perf_counter() - verification_started, 3
        )
        response.retrieval_metadata["verification_status"] = (
            "revised" if verification_result.revised_answer else "passed"
        )
    else:
        response.retrieval_metadata["verification_status"] = "disabled"

    return response
