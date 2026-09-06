"""
Agent Schemas — Models for the Agentic RAG Framework.

Phase 6 component.
Supports query decomposition, sub-question tracking, evidence confidence
scoring, and self-correction loops (PRD §11, §15).
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    tool_name: Literal[
        "hybrid_search", "graph_search", "pandas_qa",
        "web_search", "final_answer",
    ]
    query: str = Field(description="The query or code to execute with the tool.")
    reasoning: str = Field(description="Why this tool is being called.")


class AgentAction(BaseModel):
    action: ToolCall


class AgentStep(BaseModel):
    tool: str
    query: str
    observation: str
    confidence: float = Field(
        default=0.0,
        description="Confidence score (0-1) based on evidence quality from this step.",
    )
    relevance_scores: List[float] = Field(
        default_factory=list,
        description="Per-chunk relevance scores returned by the retrieval tool.",
    )


class AgentState(BaseModel):
    original_query: str
    steps: List[AgentStep] = Field(default_factory=list)
    final_answer: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)

    # ── Query Decomposition (PRD §11, §15) ──────────────────
    sub_questions: List[str] = Field(
        default_factory=list,
        description="Decomposed sub-questions for complex multi-hop queries.",
    )
    sub_answers: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of sub-question → gathered answer/evidence.",
    )
    current_sub_question: Optional[str] = Field(
        default=None,
        description="The sub-question currently being resolved.",
    )

    # ── Self-Correction (PRD §15) ───────────────────────────
    overall_confidence: float = Field(
        default=0.0,
        description="Aggregate evidence confidence across all steps (0-1).",
    )
    retry_count: int = Field(
        default=0,
        description="Number of self-correction retries performed.",
    )
