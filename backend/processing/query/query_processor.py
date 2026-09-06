"""
Query processing pipeline — rewrites, expands, and transforms user
queries for better retrieval.

Phase 2 component implementing PRD §11:
• Language Detection — detect query language (Task 3.1)
• Intent Classification — classify query intent type (Task 3.2)
• Query Rewriting — rephrase for better retrieval
• Query Expansion — add related terms
• Query Decomposition — break complex queries into sub-questions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Intent Categories ───────────────────────────────────────

INTENT_CATEGORIES = [
    "factual",       # Simple fact lookup: "What is X?"
    "analytical",    # Numerical/computation: "What is the total revenue?"
    "comparison",    # Compare entities: "How does A differ from B?"
    "definition",    # Define a term: "What is overfitting?"
    "procedural",    # How-to: "How do I configure X?"
    "visual",        # About images/charts: "What does the chart show?"
    "summarization", # Summarize content: "Summarize chapter 3"
]


@dataclass
class ProcessedQuery:
    """Result of query processing with original and rewritten forms."""
    original: str
    rewritten: Optional[str] = None
    expanded_terms: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── New fields (Tasks 3.1, 3.2) ─────────────────────────
    detected_language: str = "en"
    intent: str = "factual"

    @property
    def effective_query(self) -> str:
        """Return the best query to use for retrieval."""
        return self.rewritten or self.original

    @property
    def all_queries(self) -> List[str]:
        """Return all query variants for multi-query retrieval."""
        queries = [self.effective_query]
        if self.sub_queries:
            queries.extend(self.sub_queries)
        return queries


class QueryProcessor:
    """
    Processes user queries to improve retrieval quality.

    Uses LLM-based rewriting for better semantic matching and
    rule-based expansion for keyword coverage.
    """

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        """Lazy-load LLM for query rewriting."""
        if self._llm is None:
            from backend.generation.llm import get_llm
            self._llm = get_llm()
        return self._llm

    async def process(
        self,
        query: str,
        rewrite: bool = True,
        expand: bool = False,
        decompose: bool = False,
    ) -> ProcessedQuery:
        """
        Process a query through the full pipeline.

        Pipeline order (PRD §11):
        1. Language Detection
        2. Intent Classification
        3. Query Rewriting
        4. Query Expansion
        5. Query Decomposition

        Args:
            query: The original user query.
            rewrite: Whether to rewrite the query using LLM.
            expand: Whether to expand with related terms.
            decompose: Whether to break into sub-queries.

        Returns:
            ProcessedQuery with all query variants and metadata.
        """
        result = ProcessedQuery(original=query)

        # Step 1: Language Detection (Task 3.1)
        try:
            result.detected_language = self._detect_language(query)
            result.metadata["detected_language"] = result.detected_language
            logger.info("Detected language: %s", result.detected_language)
        except Exception as e:
            logger.warning("Language detection failed: %s", e)

        # Step 2: Intent Classification (Task 3.2)
        try:
            result.intent = await self._classify_intent(query)
            result.metadata["intent"] = result.intent
            logger.info("Classified intent: %s", result.intent)
        except Exception as e:
            logger.warning("Intent classification failed: %s", e)

        # Step 3: Query Rewriting
        if rewrite:
            try:
                result.rewritten = await self._rewrite_query(query)
                logger.info("Query rewritten: '%s' → '%s'", query[:50], result.rewritten[:50])
            except Exception as e:
                logger.warning("Query rewriting failed: %s", e)

        # Step 4: Query Expansion
        if expand:
            try:
                result.expanded_terms = await self._expand_query(query)
                logger.info("Query expanded with %d terms", len(result.expanded_terms))
            except Exception as e:
                logger.warning("Query expansion failed: %s", e)

        # Step 5: Query Decomposition
        if decompose:
            try:
                result.sub_queries = await self._decompose_query(query)
                logger.info("Query decomposed into %d sub-queries", len(result.sub_queries))
            except Exception as e:
                logger.warning("Query decomposition failed: %s", e)

        return result

    # ── Language Detection (Task 3.1 — PRD §11) ────────────

    @staticmethod
    def _detect_language(query: str) -> str:
        """
        Detect the language of the query.

        Uses a lightweight heuristic approach first, then falls back
        to the langdetect library if available.

        Returns:
            ISO 639-1 language code (e.g., 'en', 'hi', 'es').
        """
        # Try langdetect library first (fast, no LLM call)
        try:
            from langdetect import detect
            lang = detect(query)
            return lang
        except ImportError:
            pass
        except Exception:
            pass

        # Heuristic fallback: check for non-ASCII characters
        ascii_ratio = sum(1 for c in query if ord(c) < 128) / max(len(query), 1)
        if ascii_ratio > 0.9:
            return "en"

        # If we can't determine, assume English
        return "en"

    # ── Intent Classification (Task 3.2 — PRD §11) ─────────

    async def _classify_intent(self, query: str) -> str:
        """
        Classify the intent of the query using a lightweight LLM call.

        Returns one of: factual, analytical, comparison, definition,
        procedural, visual, summarization.
        """
        llm = self._get_llm()

        system_prompt = (
            "Classify the user's query intent into exactly ONE of these categories:\n"
            "- factual: Simple fact lookup (What is X? When did Y happen?)\n"
            "- analytical: Numerical/computational question (What is the total? Average?)\n"
            "- comparison: Comparing entities (How does A differ from B?)\n"
            "- definition: Define or explain a concept (What is overfitting?)\n"
            "- procedural: How-to or step-by-step (How do I configure X?)\n"
            "- visual: About images, charts, diagrams (What does the chart show?)\n"
            "- summarization: Summarize content (Summarize chapter 3)\n\n"
            "Respond with ONLY the category name, nothing else."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        try:
            response = await llm.generate(messages, temperature=0.0, max_tokens=20)
            intent = response.strip().lower().replace('"', "").replace("'", "")

            # Validate against known categories
            if intent in INTENT_CATEGORIES:
                return intent

            # Try fuzzy matching
            for cat in INTENT_CATEGORIES:
                if cat in intent:
                    return cat

            return "factual"  # Default fallback

        except Exception as e:
            logger.warning("Intent classification LLM call failed: %s", e)
            return "factual"

    # ── Query Rewriting ─────────────────────────────────────

    async def _rewrite_query(self, query: str) -> str:
        """
        Rewrite a query using the LLM for better retrieval.

        The rewrite aims to:
        - Make implicit context explicit
        - Expand abbreviations
        - Rephrase for better semantic matching
        """
        llm = self._get_llm()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a query rewriting assistant. Your job is to rewrite "
                    "user queries to improve document retrieval. Rules:\n"
                    "1. Make the query more specific and detailed\n"
                    "2. Expand abbreviations and acronyms\n"
                    "3. Add relevant context that might be implicit\n"
                    "4. Keep the core intent identical\n"
                    "5. Return ONLY the rewritten query, nothing else\n"
                    "6. Keep it concise (under 100 words)\n"
                ),
            },
            {
                "role": "user",
                "content": f"Rewrite this query for better document retrieval:\n\n{query}",
            },
        ]

        rewritten = await llm.generate(messages, temperature=0.0, max_tokens=150)
        rewritten = rewritten.strip().strip('"').strip("'")

        # Sanity check: don't use rewrite if it's too different or empty
        if not rewritten or len(rewritten) < 5:
            return query

        return rewritten

    # ── Query Expansion ─────────────────────────────────────

    async def _expand_query(self, query: str) -> List[str]:
        """
        Generate additional search terms related to the query.
        """
        llm = self._get_llm()

        messages = [
            {
                "role": "system",
                "content": (
                    "Generate 3-5 related search terms or phrases for the given query. "
                    "These should be synonyms, related concepts, or alternative phrasings "
                    "that might appear in relevant documents. "
                    "Return each term on a new line. No numbering, no explanations."
                ),
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        response = await llm.generate(messages, temperature=0.3, max_tokens=100)
        terms = [t.strip() for t in response.strip().split("\n") if t.strip()]
        return terms[:5]  # Cap at 5

    # ── Query Decomposition ─────────────────────────────────

    async def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose a complex query into simpler sub-queries.
        (Phase 6 will make this agentic.)
        """
        llm = self._get_llm()

        messages = [
            {
                "role": "system",
                "content": (
                    "Break down this complex question into 2-4 simpler sub-questions "
                    "that, when answered together, would fully answer the original question. "
                    "Return each sub-question on a new line. No numbering."
                ),
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        response = await llm.generate(messages, temperature=0.1, max_tokens=200)
        sub_queries = [q.strip() for q in response.strip().split("\n") if q.strip() and "?" in q]
        return sub_queries[:4]  # Cap at 4
