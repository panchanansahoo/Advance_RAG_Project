"""
Graph Retriever — searches Neo4j for relationships.

Phase 5 component.
"""

import logging
from typing import List, Dict, Any, Optional

from backend.generation.llm import get_llm
from backend.database.graph import get_graph_db

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Queries the Knowledge Graph for entities related to the user's question."""

    def __init__(self):
        pass

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the Neo4j graph.
        Returns a list of text descriptions representing graph paths.
        """
        try:
            driver = await get_graph_db()
        except Exception:
            # Neo4j not configured or unavailable
            return []

        # 1. Extract entities from the query
        entities = await self._extract_entities_from_query(query)
        if not entities:
            return []

        # 2. Query the graph
        results = []
        try:
            async with driver.session() as session:
                # We do a basic keyword match on node IDs
                # In production, vector search on node properties or full-text search is better.
                for entity in entities:
                    cypher = """
                    MATCH (n)-[r]-(m)
                    WHERE n.id CONTAINS $entity OR m.id CONTAINS $entity
                    RETURN n.label as n_label, n.id as n_id, type(r) as rel, m.id as m_id, m.label as m_label
                    LIMIT $limit
                    """
                    records = await session.run(cypher, entity=entity, limit=top_k)
                    
                    async for record in records:
                        desc = f"{record['n_id']} ({record['n_label']}) {record['rel']} {record['m_id']} ({record['m_label']})"
                        results.append({
                            "content": desc,
                            "score": 0.8,  # Arbitrary high score since graph relations are usually precise
                            "source": "Knowledge Graph",
                            "chunk_id": f"graph_{record['n_id']}_{record['m_id']}",
                        })
        except Exception as e:
            logger.error("Failed to query Neo4j: %s", e)

        # Deduplicate results
        unique_results = {r["content"]: r for r in results}.values()
        return list(unique_results)[:top_k]

    async def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract main entities from the user query using an LLM."""
        llm = get_llm()
        system_prompt = (
            "Extract the main named entities from the query. "
            "Return them as a comma-separated list. Only return the list, nothing else. "
            "If there are none, return 'NONE'."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        try:
            response = await llm.generate(messages, temperature=0.0, max_tokens=50)
            if response.strip() == "NONE":
                return []
            
            # Split and clean
            return [e.strip() for e in response.split(",") if e.strip()]
        except Exception as e:
            logger.warning("Failed to extract entities from query: %s", e)
            return []
