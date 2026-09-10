"""
Database Connection & Verification Script for Supabase / PostgreSQL.

Usage:
    python scripts/test_db_connection.py

This script verifies:
1. Database configuration loading from .env
2. Connectivity to PostgreSQL / Supabase
3. Automatic schema initialization (documents, chunks, conversations, messages, citations)
4. Basic read/query execution
"""

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from sqlalchemy.engine.url import make_url

from backend.config import get_settings
from backend.database.connection import _get_engine, close_db, init_db


def mask_url(url_str: str) -> str:
    """Mask password in URL for secure console logging."""
    try:
        u = make_url(url_str)
        return u.render_as_string(hide_password=True)
    except Exception:
        return "<invalid url>"


async def main():
    print("=" * 65)
    print("  Supabase / PostgreSQL Database Connection Test")
    print("=" * 65)

    settings = get_settings()
    print(f"Provider:      {settings.database_provider}")
    print(f"Target URL:    {mask_url(settings.database_url)}")
    print(f"SSL Mode:      {settings.postgres_ssl}")
    print("-" * 65)

    try:
        engine = _get_engine()
        print("1. Connecting to database engine...")
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            val = result.scalar()
            print(f"   Connection successful! (SELECT 1 returned {val})")

        print("\n2. Initializing tables (Base.metadata.create_all)...")
        await init_db()
        print("   Schema initialization completed successfully!")

        print("\n3. Verifying tables:")
        tables = ["documents", "chunks", "conversations", "messages", "citations"]
        async with engine.connect() as conn:
            for tbl in tables:
                try:
                    res = await conn.execute(text(f"SELECT COUNT(*) FROM {tbl}"))
                    count = res.scalar()
                    print(f"   - Table '{tbl}': {count} records found.")
                except Exception as table_err:
                    print(f"   - Table '{tbl}': error querying: {table_err}")

        print("\n" + "=" * 65)
        print(" SUCCESS: Database is fully connected and ready for use!")
        print("=" * 65)

    except Exception as e:
        print("\n" + "!" * 65)
        print(f" ERROR: Failed to connect to database:")
        print(f" {type(e).__name__}: {e}")
        print("!" * 65)
        print("\nTroubleshooting tips for Supabase:")
        print("1. Connection Pooler (Port 6543 / aws-0-*.pooler.supabase.com):")
        print("   If your network has issues with IPv6, use the Connection Pooler URI from:")
        print("   Supabase Dashboard -> Project Settings -> Database -> Connection string -> URI (Pooler).")
        print("2. Password special characters:")
        print("   If your database password has characters like '@', '#', or '%',")
        print("   ensure they are URL-encoded or use individual POSTGRES_PASSWORD in .env.")
        print("3. SSL Requirement:")
        print("   Supabase requires SSL. Ensure POSTGRES_SSL=require.")
        print("4. SQLite fallback:")
        print("   To run offline without PostgreSQL, set DATABASE_PROVIDER=sqlite in .env.")
        sys.exit(1)
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
