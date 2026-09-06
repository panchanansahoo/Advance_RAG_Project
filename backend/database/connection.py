"""
Async SQLAlchemy engine, session factory, and FastAPI dependency.
"""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.config import get_settings


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ── Engine & Session Factory (lazy-initialised) ────────────

_engine = None
_session_factory = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
    return _engine


def _get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=_get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


# ── FastAPI Dependency ──────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a DB session and auto-close after the request."""
    session = _get_session_factory()()
    try:
        yield session
    finally:
        await session.close()


# ── Lifecycle helpers ───────────────────────────────────────

async def init_db() -> None:
    """Create all tables (dev convenience — prefer Alembic in prod)."""
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose of the engine connection pool."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
