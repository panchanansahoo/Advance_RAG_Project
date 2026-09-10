"""
Advanced Multimodal & Agentic RAG — FastAPI Application Entry Point.

Registers all routers, middleware, and lifecycle events.
"""

from __future__ import annotations

import logging
import time
import uuid as uuid_mod
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

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

    # ── Request Timing Middleware ───────────────────────────
    class TimingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            start = time.perf_counter()
            response = await call_next(request)
            elapsed = time.perf_counter() - start
            response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
            if elapsed > 5.0:
                logger.warning(
                    "Slow request: %s %s took %.2fs",
                    request.method, request.url.path, elapsed,
                )
            return response

    app.add_middleware(TimingMiddleware)

    # ── Request ID Middleware ──────────────────────────────
    class RequestIDMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            request_id = request.headers.get("X-Request-ID", str(uuid_mod.uuid4())[:8])
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

    app.add_middleware(RequestIDMiddleware)

    # ── Rate Limiting ──────────────────────────────────────
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from backend.api.query_router import limiter

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Exception Handlers ──────────────────────────────────
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException
    from backend.utils.error_sanitizer import sanitize_error_message, format_validation_error

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        friendly_msg = format_validation_error(exc.errors())
        logger.warning("Validation error on %s %s: %s", request.method, request.url.path, exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "detail": friendly_msg,
                "error_code": "VALIDATION_ERROR",
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        friendly_msg = sanitize_error_message(exc.detail)
        logger.warning("HTTP %d on %s %s: %s", exc.status_code, request.method, request.url.path, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": friendly_msg,
                "error_code": f"HTTP_{exc.status_code}",
            },
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
        friendly_msg = sanitize_error_message(
            exc,
            default_fallback="We encountered an unexpected issue while processing your request. Please try again in a moment.",
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": friendly_msg,
                "error_code": "INTERNAL_ERROR",
            },
        )

    # ── Register Routers ───────────────────────────────────
    from backend.api.health import router as health_router
    from backend.api.auth import router as auth_router
    from backend.api.query_router import router as query_router
    from backend.api.conversations import router as conversations_router
    from backend.api.evaluation import router as evaluation_router
    from backend.ingestion.router import router as ingestion_router

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(ingestion_router)
    app.include_router(query_router)
    app.include_router(conversations_router)
    app.include_router(evaluation_router)

    # ── Root & Static Mounts ───────────────────────────────
    import os
    vanilla_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vanilla_frontend")
    if os.path.isdir(vanilla_dir):
        app.mount("/vanilla", StaticFiles(directory=vanilla_dir, html=True), name="vanilla_frontend")
        app.mount("/", StaticFiles(directory=vanilla_dir, html=True), name="frontend")
    else:
        @app.get("/")
        async def root():
            return {
                "name": "Advanced Multimodal Agentic RAG API",
                "docs": "/docs",
                "health": "/api/v1/health",
                "status": "online",
            }

    return app


# ── App Instance ────────────────────────────────────────────
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

