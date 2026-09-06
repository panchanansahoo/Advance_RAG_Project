"""
Neo4j Graph Database Connection Management.

Phase 5 component.
"""

import logging
from typing import Optional
from neo4j import GraphDatabase, AsyncGraphDatabase, AsyncDriver

from backend.config import get_settings

logger = logging.getLogger(__name__)

_driver: Optional[AsyncDriver] = None


async def get_graph_db() -> AsyncDriver:
    """
    Get the Neo4j AsyncDriver instance.
    Raises ValueError if Neo4j is not configured or unavailable.
    """
    global _driver
    settings = get_settings()

    if _driver is None:
        try:
            _driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )
            # Verify connectivity
            await _driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", settings.neo4j_uri)
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", e)
            _driver = None
            raise ValueError(f"Neo4j connection failed: {e}")

    return _driver


async def close_graph_db():
    """Close the Neo4j driver connection."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
        logger.info("Closed Neo4j connection.")
