"""
Query Router — determines the best execution path for a user's question.

Phase 4 component: routes between standard RAG (Text), Structured Data (Pandas/SQL),
Visual (VLM/multimodal RAG), and Agentic (complex multi-hop) paths.

Implements all four PRD §11 routes:
  Text → RAG | Structured data → Pandas/SQL | Visual → VLM | Complex → Agentic RAG
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field
from backend.generation.llm import get_llm
from backend.config import get_settings

logger = logging.getLogger(__name__)


class RouteType(str, Enum):
    RAG = "rag"
    STRUCTURED_DATA = "structured_data"
    VISUAL = "visual"
    AGENTIC = "agentic"


class RoutingDecision(BaseModel):
    route: RouteType = Field(description="The selected execution route.")
    reasoning: str = Field(description="Why this route was selected.")


class QueryRouter:
    """
    Classifies a user query to determine if it should use standard vector RAG,
    structured data agent, visual/multimodal RAG, or the agentic orchestrator.
    """

    def __init__(self):
        self.settings = get_settings()

    async def route_query(
        self,
        query: str,
        available_document_types: Optional[List[str]] = None,
    ) -> RouteType:
        """
        Determine the execution route for a given query.

        Args:
            query: The user's question.
            available_document_types: List of document extensions in context (e.g. ['.pdf', '.csv']).

        Returns:
            RouteType indicating the execution path.
        """
        if not self.settings.query_routing_enabled:
            return RouteType.RAG

        # ── Heuristic pre-checks ────────────────────────────
        has_structured = False
        has_visual = False
        if available_document_types:
            has_structured = any(
                ext in [".csv", ".xlsx", ".xls"]
                for ext in available_document_types
            )
            has_visual = any(
                ext in [".png", ".jpg", ".jpeg"]
                for ext in available_document_types
            )

        # If only text docs and no complex signals, fast-path to RAG
        if available_document_types is not None and not has_structured and not has_visual:
            # Still check for complex multi-hop queries that need agentic
            if not self._looks_complex(query):
                return RouteType.RAG

        # Heuristic fast-path: structured data keywords + structured files available
        if has_structured and self._looks_structured(query):
            logger.info("Fast-path routing to structured_data (heuristic match)")
            return RouteType.STRUCTURED_DATA

        # Heuristic fast-path: visual keywords + image files available
        if has_visual and self._looks_visual(query):
            logger.info("Fast-path routing to visual (heuristic match)")
            return RouteType.VISUAL

        try:
            llm = get_llm()

            system_prompt = (
                "You are an intelligent query router for a document QA system. "
                "Your job is to classify the user's question into one of the following routes:\n\n"
                "1. 'structured_data': Select this if the user is asking numerical, analytical, "
                "or aggregation questions that require exact computation over tabular data (like CSVs or Excel). "
                "Examples: 'What is the total revenue in Q3?', 'What is the average age?', 'How many rows match X?'.\n"
                "2. 'visual': Select this if the user is asking about images, charts, diagrams, "
                "screenshots, or visual content that requires visual understanding. "
                "Examples: 'What does the chart on page 5 show?', 'Describe the architecture diagram', "
                "'What trend is visible in the graph?'.\n"
                "3. 'agentic': Select this if the question is complex, requires multi-hop reasoning, "
                "comparison across multiple documents, or combines different data types. "
                "Examples: 'Compare the results from Report A and Report B', "
                "'What caused the revenue change mentioned in the financial report and how does it relate to the HR data?', "
                "'Research the topic across all uploaded documents and provide a comprehensive analysis'.\n"
                "4. 'rag': Select this for all other questions — conceptual questions, summaries, "
                "definitions, or straightforward questions about text documents.\n\n"
                f"Context: Structured data files available? {'Yes' if has_structured else 'No'}. "
                f"Image files available? {'Yes' if has_visual else 'No'}.\n\n"
                "Respond in valid JSON matching the following schema:\n"
                '{"route": "rag" | "structured_data" | "visual" | "agentic", "reasoning": "your reasoning here"}'
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"},
            ]

            response_text = await llm.generate(
                messages,
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )

            import json

            data = json.loads(response_text)
            decision = RoutingDecision(**data)

            # Validate: don't route to structured_data if no structured files
            if decision.route == RouteType.STRUCTURED_DATA and not has_structured:
                logger.info("Overriding structured_data route — no structured files available")
                decision.route = RouteType.RAG

            # Validate: don't route to visual if no image files
            if decision.route == RouteType.VISUAL and not has_visual:
                logger.info("Overriding visual route — no image files available")
                decision.route = RouteType.RAG

            logger.info("Query routed to %s: %s", decision.route, decision.reasoning)
            return decision.route

        except Exception as e:
            logger.warning("Query routing failed, defaulting to RAG: %s", e)
            return RouteType.RAG

    @staticmethod
    def _looks_complex(query: str) -> bool:
        """Quick heuristic check for multi-hop or comparison queries."""
        lower = query.lower()
        complexity_signals = [
            "compare", "comparison", "versus", " vs ",
            "difference between", "relate", "relationship",
            "across", "multiple", "both", "each",
            "step by step", "research", "analyze",
            "how does * relate to", "what caused",
        ]
        return any(signal in lower for signal in complexity_signals)

    @staticmethod
    def _looks_structured(query: str) -> bool:
        """Quick heuristic for numerical/analytical queries suited to pandas."""
        lower = query.lower()
        structured_signals = [
            "total", "average", "mean", "median", "sum", "count",
            "how many", "percentage", "max", "min", "highest", "lowest",
            "top ", "bottom ", "rank", "sort", "group by", "aggregate",
            "calculate", "compute", "spreadsheet", "column",
        ]
        return any(signal in lower for signal in structured_signals)

    @staticmethod
    def _looks_visual(query: str) -> bool:
        """Quick heuristic for visual/image-related queries."""
        lower = query.lower()
        visual_signals = [
            "image", "picture", "photo", "chart", "graph", "diagram",
            "screenshot", "visual", "figure", "drawing", "illustration",
            "what does the", "describe the", "show me", "look like",
        ]
        return any(signal in lower for signal in visual_signals)


# ── Singleton accessor (Task 4.1) ───────────────────────────
_query_router: Optional[QueryRouter] = None


def get_query_router() -> QueryRouter:
    """Return a cached QueryRouter singleton."""
    global _query_router
    if _query_router is None:
        _query_router = QueryRouter()
    return _query_router


