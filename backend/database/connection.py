"""
Async SQLAlchemy engine, session factory, and FastAPI dependency.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.config import get_settings

logger = logging.getLogger(__name__)


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
        url = settings.database_url
        engine_kwargs = {
            "echo": settings.debug,
        }

        if "asyncpg" in url:
            engine_kwargs["pool_pre_ping"] = True
            engine_kwargs["pool_size"] = 10
            engine_kwargs["max_overflow"] = 20
            engine_kwargs["pool_recycle"] = 3600  # Recycle connections after 1 hour
            engine_kwargs["pool_timeout"] = 30  # Max wait time for a connection

            connect_args = {}

            # Supabase pooler / PgBouncer transaction mode requires disabling prepared statements
            if "pooler.supabase.com" in url or ":6543" in url:
                connect_args["prepared_statement_cache_size"] = 0

            # Supabase / cloud PostgreSQL SSL requirements
            is_supabase = "supabase.co" in url or "supabase.com" in url
            ssl_required = (
                is_supabase
                or (
                    settings.postgres_ssl
                    and settings.postgres_ssl.lower()
                    in ("require", "true", "verify-ca", "verify-full")
                )
                or "ssl=require" in url
            )
            if ssl_required:
                connect_args["ssl"] = "require"

            if connect_args:
                engine_kwargs["connect_args"] = connect_args

        _engine = create_async_engine(url, **engine_kwargs)
        logger.info(
            "Database engine created: %s (pool_size=%s, max_overflow=%s)",
            "asyncpg" if "asyncpg" in url else "aiosqlite",
            engine_kwargs.get("pool_size", "default"),
            engine_kwargs.get("max_overflow", "default"),
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
    import backend.database.models  # noqa: F401

    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Keep existing development databases compatible without requiring Alembic.
        from sqlalchemy import inspect, text

        columns = await conn.run_sync(
            lambda sync_conn: {column["name"] for column in inspect(sync_conn).get_columns("conversations")}
        )
        if "owner_key" not in columns:
            await conn.execute(text("ALTER TABLE conversations ADD COLUMN owner_key VARCHAR(512)"))
        document_columns = await conn.run_sync(
            lambda sync_conn: {column["name"] for column in inspect(sync_conn).get_columns("documents")}
        )
        if "owner_key" not in document_columns:
            await conn.execute(text("ALTER TABLE documents ADD COLUMN owner_key VARCHAR(512)"))


async def close_db() -> None:
    """Dispose of the engine connection pool."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
