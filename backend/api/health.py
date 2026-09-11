"""Health check endpoint with comprehensive dependency verification."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])

# Track server start time for uptime reporting
_start_time = time.monotonic()


@router.get("/api/v1/health/live")
async def liveness_check():
    """Confirm that the API process is running."""
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    System health check with comprehensive dependency verification.

    Checks:
    - API responsiveness
    - Database connectivity
    - Vector store connectivity
    - BM25 index status
    - Embedding model status
    """
    checks = {
        "api": "ok",
        "database": "unknown",
        "vector_store": "unknown",
        "bm25_index": "unknown",
        "embedding_model": "unknown",
    }
    overall = "healthy"

    # ── Database Check ──────────────────────────────────────
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = "error"
        overall = "degraded"
        logger.warning("Health check — database unreachable: %s", e)

    # ── Vector Store Check ──────────────────────────────────
    try:
        from backend.retrieval.vector_store import get_vector_store

        vs = get_vector_store()
        # ChromaDB and Qdrant both support a lightweight operation
        # to verify connectivity without heavy queries.
        if hasattr(vs, "_client"):
            # ChromaDB: heartbeat or list_collections
            vs._client.heartbeat()
            checks["vector_store"] = "ok"
        elif hasattr(vs, "client"):
            # Qdrant: simple info call
            vs.client.get_collections()
            checks["vector_store"] = "ok"
        else:
            checks["vector_store"] = "ok"
    except Exception as e:
        checks["vector_store"] = "error"
        overall = "degraded"
        logger.warning("Health check — vector store unreachable: %s", e)

    # ── BM25 Index Check ────────────────────────────────────
    try:
        from backend.retrieval.bm25_search import get_bm25_index

        bm25 = get_bm25_index()
        if bm25.is_built:
            checks["bm25_index"] = f"ok ({bm25.size} chunks)"
        else:
            checks["bm25_index"] = "not_built"
    except Exception as e:
        checks["bm25_index"] = "error"
        logger.warning("Health check — BM25 index error: %s", e)

    # ── Embedding Model Check ───────────────────────────────
    try:
        from backend.embeddings import get_embedding_provider

        emb = get_embedding_provider()
        checks["embedding_model"] = f"ok ({emb.model_name})"
    except Exception as e:
        checks["embedding_model"] = "error"
        overall = "degraded"
        logger.warning("Health check — embedding model error: %s", e)

    # ── Compute uptime ──────────────────────────────────────
    uptime_seconds = int(time.monotonic() - _start_time)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours}h {minutes}m {seconds}s"

    return {
        "status": overall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "Advanced RAG API",
        "version": "1.0.0",
        "uptime": uptime_str,
        "checks": checks,
    }


@router.get("/api/v1/health/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """Return a non-2xx response when required dependencies are degraded."""
    details = await health_check(db)
    if details["status"] != "healthy":
        return JSONResponse(status_code=503, content=details)
    return details
