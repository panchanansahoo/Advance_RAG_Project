"""
Graph Extractor — extracts entities and relationships from text for Knowledge Graph.

Phase 5 component.
"""

import logging
import json
from typing import List, Dict, Any

from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.generation.llm import get_llm
from backend.database.graph import get_graph_db

logger = logging.getLogger(__name__)


class Node(BaseModel):
    id: str = Field(description="Unique identifier for the node (e.g. 'Apple Inc.')")
    label: str = Field(description="Node type (e.g. 'Organization', 'Person', 'Concept', 'Location', 'Product')")


class Edge(BaseModel):
    source: str = Field(description="ID of the source node")
    target: str = Field(description="ID of the target node")
    type: str = Field(description="Relationship type (e.g. 'FOUNDED', 'LOCATED_IN', 'RELATED_TO', 'HAS_PART')")


class GraphExtractionResult(BaseModel):
    nodes: List[Node] = Field(description="Extracted nodes", default_factory=list)
    edges: List[Edge] = Field(description="Extracted edges", default_factory=list)


class GraphExtractor:
    """Extracts entities and relationships from text to build a Knowledge Graph."""

    def __init__(self):
        self.settings = get_settings()

    async def extract_and_store(self, text: str, document_id: str) -> None:
        """
        Extracts a graph from the given text and stores it in Neo4j.
        Associates all nodes with the given document_id.
        """
        if not self.settings.graph_extraction_enabled:
            return

        graph_data = await self._extract_graph_from_text(text)
        if graph_data and (graph_data.nodes or graph_data.edges):
            await self._store_in_neo4j(graph_data, document_id)

    async def _extract_graph_from_text(self, text: str) -> GraphExtractionResult:
        """Use LLM to extract nodes and edges."""
        llm = get_llm()
        system_prompt = (
            "You are an expert knowledge graph extractor. Your task is to extract entities (nodes) "
            "and relationships (edges) from the provided text.\n\n"
            "Rules:\n"
            "1. Node 'id' must be the exact name of the entity (e.g. 'Tim Cook').\n"
            "2. Node 'label' must be a generic category (e.g. 'Person', 'Organization', 'Location', 'Product', 'Concept').\n"
            "3. Edge 'source' and 'target' must match the Node 'id's exactly.\n"
            "4. Edge 'type' must be an uppercase verb phrase (e.g. 'WORKS_FOR', 'LOCATED_IN', 'IS_A', 'RELATES_TO').\n"
            "5. Return the result in valid JSON matching the following schema:\n"
            '{\n'
            '  "nodes": [{"id": "...", "label": "..."}],\n'
            '  "edges": [{"source": "...", "target": "...", "type": "..."}]\n'
            '}\n'
            "Extract only the most important factual relationships. Do not extract stop words as entities."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Text to extract:\n{text[:3000]}"}  # Limit text to avoid huge context costs
        ]

        try:
            response_text = await llm.generate(
                messages,
                temperature=0.0,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response_text)
            return GraphExtractionResult(**data)
            
        except Exception as e:
            logger.warning("Graph extraction failed (returning empty graph): %s", e)
            return GraphExtractionResult(nodes=[], edges=[])

    async def _store_in_neo4j(self, graph_data: GraphExtractionResult, document_id: str) -> None:
        """Store the extracted graph nodes and edges in Neo4j using APOC or direct Cypher."""
        try:
            driver = await get_graph_db()
            
            # Use a session to run transactions
            async with driver.session() as session:
                # 1. Merge Nodes
                for node in graph_data.nodes:
                    # Sanitize label (alphanumeric only)
                    label = "".join(c for c in node.label if c.isalnum()) or "Entity"
                    
                    query = f"""
                    MERGE (n:`{label}` {{id: $id}})
                    ON CREATE SET n.document_id = $doc_id
                    """
                    await session.run(query, id=node.id, doc_id=document_id)

                # 2. Merge Edges
                for edge in graph_data.edges:
                    # Sanitize rel_type
                    rel_type = "".join(c for c in edge.type if c.isalnum() or c == '_').upper() or "RELATED_TO"
                    
                    # We don't know the labels of source and target during edge insertion easily without lookup, 
                    # so we match on ID across all nodes. This might be slow on huge graphs without index.
                    # Creating a global index on `id` across all nodes is recommended.
                    query = f"""
                    MATCH (source {{id: $source_id}})
                    MATCH (target {{id: $target_id}})
                    MERGE (source)-[r:`{rel_type}`]->(target)
                    """
                    await session.run(query, source_id=edge.source, target_id=edge.target)

            logger.info("Stored %d nodes and %d edges for document %s", len(graph_data.nodes), len(graph_data.edges), document_id)
            
        except Exception as e:
            logger.error("Failed to store graph in Neo4j: %s", e)
