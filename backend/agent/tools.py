"""
Agent Tools — Wrappers around retrieval and execution services.

Phase 6 component.
Returns structured observations with relevance scores for the agent's
self-correction loop (PRD §15).
"""

import logging
from typing import List, Dict, Any, Tuple
from uuid import UUID

from backend.retrieval.service import RetrievalService
from backend.retrieval.graph_retriever import GraphRetriever
from backend.agent.pandas_agent import PandasAgent

logger = logging.getLogger(__name__)


class HybridSearchTool:
    """Wrapper around Vector + BM25 + Reranker pipeline."""

    def __init__(self, retrieval_service: Optional[RetrievalService] = None):
        self.retrieval_service = retrieval_service or RetrievalService()

    async def execute(
        self, query: str, document_ids: List[UUID] = None
    ) -> Tuple[str, List[Dict[str, Any]], List[float]]:
        """
        Runs hybrid search and returns a text summary, citation list,
        and per-chunk relevance scores for evidence quality assessment.

        Returns:
            Tuple of (observation_text, citations, relevance_scores)
        """
        try:
            chunks = await self.retrieval_service.retrieve_chunks(
                query=query,
                top_k=5,
                document_ids=document_ids,
            )

            if not chunks:
                return "No relevant text documents found.", [], []

            # Build observation with embedded relevance scores
            observation_parts = []
            for c in chunks:
                score_tag = f"[Relevance: {c.score:.3f}]"
                observation_parts.append(
                    f"{score_tag} [Doc: {c.source_filename}] {c.content}"
                )
            observation = "\n\n".join(observation_parts)

            # Extract per-chunk scores for the self-correction loop
            relevance_scores = [c.score for c in chunks]

            citations = []
            for c in chunks:
                citations.append(
                    {
                        "document_id": str(c.document_id),
                        "document_name": c.source_filename or "Unknown",
                        "chunk_id": str(c.chunk_id),
                        "content_snippet": c.content[:200],
                        "relevance_score": c.score,
                    }
                )

            return observation, citations, relevance_scores

        except Exception as e:
            logger.error("HybridSearchTool failed: %s", e)
            return f"Error executing hybrid search: {str(e)}", [], []


class GraphSearchTool:
    """Wrapper around Neo4j Graph Retriever."""

    def __init__(self, graph_retriever: Optional[GraphRetriever] = None):
        self.graph_retriever = graph_retriever or GraphRetriever()

    async def execute(self, query: str) -> Tuple[str, List[float]]:
        """
        Runs graph search and returns entity relationships with scores.

        Returns:
            Tuple of (observation_text, relevance_scores)
        """
        try:
            results = await self.graph_retriever.search(query=query, top_k=5)
            if not results:
                return "No relevant graph connections found.", []

            observation_parts = []
            scores = []
            for r in results:
                score = r.get("score", 0.0)
                scores.append(score)
                observation_parts.append(
                    f"[Relevance: {score:.3f}] {r['content']}"
                )
            return "\n".join(observation_parts), scores
        except Exception as e:
            logger.error("GraphSearchTool failed: %s", e)
            return f"Error executing graph search: {str(e)}", []


class PandasQATool:
    """Wrapper around PandasAgent for structured data."""

    def __init__(self, pandas_agent: Optional[PandasAgent] = None):
        self.pandas_agent = pandas_agent or PandasAgent()

    async def execute(self, query: str, file_paths: List[str]) -> str:
        """Runs pandas agent against tabular files."""
        if not file_paths:
            return "No structured data files (.csv, .xlsx) available for analysis."

        try:
            result = await self.pandas_agent.run(query=query, file_paths=file_paths)
            return result
        except Exception as e:
            logger.error("PandasQATool failed: %s", e)
            return f"Error executing pandas agent: {str(e)}"

