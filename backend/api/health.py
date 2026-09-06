"""Health check endpoint."""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/api/v1/health")
async def health_check():
    """System health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Advanced RAG API",
        "version": "1.0.0",
    }
