"""
Advanced Multimodal & Agentic RAG — FastAPI Application Entry Point.

Registers all routers, middleware, and lifecycle events.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import get_settings

# ── Logging Setup ───────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ───────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("  %s — Starting up", settings.app_name)
    logger.info("  Environment: %s", settings.app_env)
    logger.info("  LLM Provider: %s (%s)", settings.llm_provider, settings.llm_model)
    logger.info("  Embedding Provider: %s (%s)", settings.embedding_provider, settings.embedding_model)
    logger.info("  Vector DB: %s", settings.vector_db_provider)
    logger.info("=" * 60)

    # Initialize database tables
    try:
        from backend.database.connection import init_db
        await init_db()
        logger.info("Database tables initialized")
    except Exception as e:
        logger.warning("Database initialization skipped: %s", e)

    # Initialize vector store
    try:
        from backend.embeddings import get_embedding_provider
        from backend.retrieval.vector_store import get_vector_store

        vector_store = get_vector_store()
        embedding_provider = get_embedding_provider()
        await vector_store.initialize(
            settings.qdrant_collection,
            embedding_provider.dimension,
        )
        logger.info("Vector store initialized")
    except Exception as e:
        logger.warning("Vector store initialization skipped: %s", e)

    # Load persisted BM25 index (Task 2.4)
    try:
        from backend.retrieval.bm25_search import get_bm25_index

        bm25_index = get_bm25_index()
        if bm25_index.load_from_disk():
            logger.info("BM25 index loaded from disk (%d chunks)", bm25_index.size)
        else:
            logger.info("No persisted BM25 index found — will build on first ingestion")
    except Exception as e:
        logger.warning("BM25 index loading skipped: %s", e)

    yield

    # Shutdown
    logger.info("Shutting down...")
    try:
        from backend.database.connection import close_db
        from backend.database.graph import close_graph_db
        await close_graph_db()
        await close_db()
    except Exception:
        pass


# ── Application ─────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "An Advanced Multimodal and Agentic Retrieval-Augmented Generation "
            "Framework for Evidence-Grounded Multi-Document Question Answering."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS Middleware ─────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global Exception Handler ────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An internal server error occurred",
                "error_code": "INTERNAL_ERROR",
            },
        )

    # ── Register Routers ───────────────────────────────────
    from backend.api.health import router as health_router
    from backend.api.query_router import router as query_router
    from backend.api.conversations import router as conversations_router
    from backend.api.evaluation import router as evaluation_router
    from backend.ingestion.router import router as ingestion_router

    app.include_router(health_router)
    app.include_router(ingestion_router)
    app.include_router(query_router)
    app.include_router(conversations_router)
    app.include_router(evaluation_router)

    # ── Serve Frontend ─────────────────────────────────────
    # Mount frontend static files (served at root)
    import os
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
    if os.path.isdir(frontend_dir):
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app


# ── App Instance ────────────────────────────────────────────
app = create_app()
