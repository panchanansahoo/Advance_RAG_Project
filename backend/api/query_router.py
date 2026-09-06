"""
Query API router — the main QA endpoint.

Handles all 4 route types (PRD §11):
  RAG | Structured Data | Visual | Agentic

Task 1.3: Routes visual and agentic queries to their respective handlers.
Task 1.4: Passes full retrieved chunks (not truncated snippets) to verification.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database.connection import get_db
from backend.database.models import Document
from backend.database.repositories.conversation_repo import ConversationRepository
from backend.generation.service import GenerationService
from backend.schemas.queries import QueryRequest, QueryResponse
from backend.routing.query_router import QueryRouter, RouteType
from backend.agent.pandas_agent import PandasAgent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    """
    Ask a question and get a grounded answer with citations.
    Phase 4: Routes query to RAG, Pandas Agent, Visual, or Agentic based on context and intent.
    Phase 6: Agentic RAG with iterative tool use, decomposition, and self-correction.
    Phase 7: Verification and contradiction detection.
    §20: Conversation memory — loads and saves messages.
    """
    try:
        # 0. Load conversation history (if conversation_id provided)
        conversation_messages = []
        conv_repo = ConversationRepository(db)

        if request.conversation_id:
            try:
                conversation_messages = await conv_repo.get_recent_messages(
                    request.conversation_id, limit=10
                )
            except Exception as e:
                logger.warning("Failed to load conversation history: %s", e)
        else:
            # Auto-create a conversation if none provided
            try:
                conv = await conv_repo.create_conversation(
                    title=request.query[:80]
                )
                request = request.model_copy(
                    update={"conversation_id": conv.id}
                )
            except Exception as e:
                logger.warning("Failed to auto-create conversation: %s", e)

        # 1. Fetch document extensions and paths if document_ids are provided
        available_extensions = []
        file_paths = []
        image_file_paths = []

        if request.document_ids:
            result = await db.execute(
                select(Document).where(Document.id.in_(request.document_ids))
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
        if request.force_route:
            try:
                route = RouteType(request.force_route.lower())
                logger.info("Per-request forced route: %s", route)
            except ValueError:
                logger.warning(
                    "Unknown forced route '%s', falling back to adaptive routing",
                    request.force_route,
                )
        elif request.use_agent is True:
            route = RouteType.AGENTIC
            logger.info("Per-request forced agentic RAG via use_agent=True")
        elif request.use_agent is False:
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
            route = await router_svc.route_query(request.query, available_extensions)
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
                request, file_paths, settings, db
            )

        elif route == RouteType.STRUCTURED_DATA and file_paths:
            response = await _handle_structured_data(request, file_paths)

        elif route == RouteType.VISUAL and image_file_paths:
            response = await _handle_visual(request, image_file_paths, conversation_messages)

        else:
            # Default: standard RAG pipeline
            response = await _handle_rag(
                request, conversation_messages, settings
            )

        # 4. Save conversation messages (§20)
        if request.conversation_id:
            try:
                # Save user message
                await conv_repo.add_message(
                    conversation_id=request.conversation_id,
                    role="user",
                    content=request.query,
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
                    conversation_id=request.conversation_id,
                    role="assistant",
                    content=response.answer,
                    citations=citation_data,
                    metadata=response.retrieval_metadata,
                )
            except Exception as e:
                logger.warning("Failed to save conversation messages: %s", e)

        return response

    except Exception as e:
        logger.error("Query failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}",
        )


# ── Route Handlers ──────────────────────────────────────────


async def _handle_agentic(
    request: QueryRequest,
    file_paths: list,
    settings,
    db: AsyncSession,
) -> QueryResponse:
    """Handle agentic RAG with decomposition and self-correction (Phase 6)."""
    logger.info("Executing via Agentic RAG Orchestrator")
    from backend.agent.orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator()
    state = await orchestrator.run(
        query=request.query,
        document_ids=request.document_ids,
        structured_file_paths=file_paths,
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
) -> QueryResponse:
    """Handle standard RAG pipeline (Phase 1-2)."""
    logger.info("Executing via standard RAG pipeline")
    from backend.generation.service import get_generation_service
    from backend.verification.service import get_verification_service

    service = get_generation_service()
    response = await service.answer(
        request,
        conversation_messages=conversation_messages,
    )
    response.retrieval_metadata["route"] = "rag"

    # Phase 7: Reliability & Verification
    if settings.verification_enabled:
        verifier = get_verification_service()
        verification_result = await verifier.verify(
            request.query, response.answer, response.citations
        )
        response.answer = verification_result.revised_answer or response.answer

    return response
