"""
Agent Orchestrator — iterative planning and execution loop for Agentic RAG.

Phase 6 component.

Implements:
- Query decomposition for multi-hop questions (PRD §11)
- Self-correction loop with evidence confidence evaluation (PRD §15)
- Intelligent observation truncation preserving sentence boundaries (fix for 2.1)
- Sub-question tracking and synthesis
- Loop-detection guardrail to prevent infinite spinning
- Improved token budget estimation
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
from uuid import UUID

from backend.config import get_settings
from backend.generation.llm import get_llm
from backend.schemas.agent import AgentState, AgentStep, AgentAction, ToolCall
from backend.agent.tools import HybridSearchTool, GraphSearchTool, PandasQATool

logger = logging.getLogger(__name__)

# ── Evidence quality thresholds ─────────────────────────────
CONFIDENCE_SUFFICIENT = 0.45   # Above this → evidence is good enough
CONFIDENCE_LOW = 0.25          # Below this → trigger self-correction
MAX_RETRIES = 2                # Max self-correction retries per sub-question


def _estimate_tokens(text: str) -> int:
    """Estimate token count. Uses tiktoken if available, else falls back to word-based."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: ~0.75 words per token for English text
        return int(len(text.split()) * 1.33)


class AgentOrchestrator:
    """Iterative Agent Loop for complex question answering."""

    def __init__(self):
        self.settings = get_settings()
        self.hybrid_tool = HybridSearchTool()
        self.graph_tool = GraphSearchTool()
        self.pandas_tool = PandasQATool()
        self._token_usage = 0
        self._token_budget = self.settings.agent_token_budget

    async def run(
        self,
        query: str,
        document_ids: Optional[List[UUID]] = None,
        structured_file_paths: Optional[List[str]] = None,
        conversation_messages: Optional[list] = None,
        owner_key: Optional[str] = None,
    ) -> AgentState:
        """
        Execute the agent loop until a final answer is generated or max iterations reached.

        Pipeline:
        1. Decompose complex query into sub-questions (if applicable)
        2. For each sub-question (or the original query):
           a. Plan next action
           b. Execute tool
           c. Evaluate evidence confidence
           d. Self-correct if confidence is too low (rewrite & re-search)
        3. Synthesize final answer from all sub-answers
        """
        self._token_usage = 0  # Reset per-run
        run_start = time.perf_counter()
        state = AgentState(original_query=query)
        max_iters = self.settings.agent_max_iterations

        # ── Step 0: Query Decomposition ─────────────────────
        state.sub_questions = await self._decompose_query(query)
        if state.sub_questions:
            logger.info(
                "Query decomposed into %d sub-questions: %s",
                len(state.sub_questions),
                state.sub_questions,
            )
        else:
            # Treat the original query as the sole "sub-question"
            state.sub_questions = [query]

        # ── Iterate over sub-questions ──────────────────────
        global_iter = 0
        # Loop-detection: track recent (tool, query) pairs to detect spinning
        recent_actions: List[tuple] = []

        for sub_q in state.sub_questions:
            state.current_sub_question = sub_q
            state.retry_count = 0
            sub_q_resolved = False

            for _ in range(max_iters):
                if global_iter >= max_iters * len(state.sub_questions):
                    logger.warning("Global iteration limit reached. Forcing final answer.")
                    break
                global_iter += 1

                # Token budget guard
                if self._token_usage >= self._token_budget:
                    logger.warning(
                        "Token budget exhausted (%d/%d). Forcing final answer.",
                        self._token_usage,
                        self._token_budget,
                    )
                    break

                step_start = time.perf_counter()
                logger.info(
                    "Agent iteration %d | Sub-question: '%s' | Tokens used: %d/%d",
                    global_iter,
                    sub_q[:80],
                    self._token_usage,
                    self._token_budget,
                )

                # 1. Plan next action
                action = await self._plan_next_action(state, structured_file_paths, conversation_messages)

                if not action or action.action.tool_name == "final_answer":
                    # Sub-question resolved — record the sub-answer
                    sub_answer = action.action.query if action else ""
                    state.sub_answers[sub_q] = sub_answer
                    sub_q_resolved = True
                    break

                # Loop-detection guardrail: if same (tool, query) appears 3 times, force final answer
                action_key = (action.action.tool_name, action.action.query.strip().lower())
                recent_actions.append(action_key)
                if recent_actions.count(action_key) >= 3:
                    logger.warning(
                        "Loop detected: tool=%s, query='%s' repeated 3 times. Forcing final answer.",
                        action.action.tool_name,
                        action.action.query[:60],
                    )
                    evidence_summary = "\n".join(
                        s.observation[:500] for s in state.steps[-3:]
                    )
                    state.sub_answers[sub_q] = f"[Partial evidence] {evidence_summary}"
                    sub_q_resolved = True
                    break

                # 2. Execute tool
                observation, new_citations, relevance_scores = await self._execute_tool(
                    action.action.tool_name,
                    action.action.query,
                    document_ids,
                    structured_file_paths,
                    owner_key,
                )

                # 3. Compute step confidence
                step_confidence = self._compute_confidence(relevance_scores)

                # 4. Update state
                step_elapsed = time.perf_counter() - step_start
                step = AgentStep(
                    tool=action.action.tool_name,
                    query=action.action.query,
                    observation=observation,
                    confidence=step_confidence,
                    relevance_scores=relevance_scores,
                )
                state.steps.append(step)
                if new_citations:
                    state.citations.extend(new_citations)

                logger.info(
                    "Step %d completed in %.2fs | tool=%s | confidence=%.3f",
                    global_iter, step_elapsed, action.action.tool_name, step_confidence,
                )

                # 5. Self-correction check (PRD §15)
                if step_confidence < CONFIDENCE_LOW and state.retry_count < MAX_RETRIES:
                    logger.info(
                        "Low confidence (%.3f < %.3f). Triggering self-correction (retry %d/%d).",
                        step_confidence,
                        CONFIDENCE_LOW,
                        state.retry_count + 1,
                        MAX_RETRIES,
                    )
                    state.retry_count += 1
                    # The next planning call will see the low-confidence observation
                    # and the system prompt instructs it to rewrite the query
                    continue

            # If sub-question wasn't resolved, record what we have
            if not sub_q_resolved:
                evidence_summary = "\n".join(
                    s.observation[:500] for s in state.steps[-3:]
                )
                state.sub_answers[sub_q] = f"[Partial evidence] {evidence_summary}"

        # ── Update overall confidence ───────────────────────
        all_confidences = [s.confidence for s in state.steps if s.confidence > 0]
        state.overall_confidence = (
            sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        )

        # ── Final answer synthesis ──────────────────────────
        state.current_sub_question = None
        state.final_answer = await self._generate_final_answer(state)

        total_elapsed = time.perf_counter() - run_start
        logger.info(
            "Agent run completed in %.2fs | %d steps | confidence=%.3f | tokens=%d/%d",
            total_elapsed, len(state.steps), state.overall_confidence,
            self._token_usage, self._token_budget,
        )

        return state

    # ── Query Decomposition ─────────────────────────────────

    async def _decompose_query(self, query: str) -> List[str]:
        """
        Use LLM to determine if a query needs decomposition and, if so,
        break it into simpler sub-questions.

        Returns empty list if the query is simple enough to answer directly.
        """
        llm = get_llm()

        system_prompt = (
            "You are a query analysis assistant. Analyze the user's question and determine "
            "if it requires decomposition into simpler sub-questions.\n\n"
            "A query NEEDS decomposition if it:\n"
            "- Asks about multiple entities or documents (comparison questions)\n"
            "- Requires multi-hop reasoning (A relates to B, B relates to C)\n"
            "- Contains multiple distinct sub-questions joined by 'and', 'also', etc.\n"
            "- Requires both textual and numerical/analytical reasoning\n\n"
            "A query does NOT need decomposition if it:\n"
            "- Is a single factual question\n"
            "- Asks about one concept or entity\n"
            "- Is a simple definition or explanation request\n\n"
            "Respond in JSON:\n"
            '{"needs_decomposition": true/false, "sub_questions": ["q1", "q2", ...]}\n'
            "If needs_decomposition is false, return an empty sub_questions array."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"},
        ]

        try:
            response = await llm.generate(
                messages,
                temperature=0.0,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            self._token_usage += _estimate_tokens(response) + _estimate_tokens(query)
            data = json.loads(response)
            if data.get("needs_decomposition") and data.get("sub_questions"):
                return data["sub_questions"][:4]  # Cap at 4 sub-questions
            return []
        except Exception as e:
            logger.warning("Query decomposition failed: %s", e)
            return []

    # ── Planning ────────────────────────────────────────────

    async def _plan_next_action(
        self, state: AgentState, structured_file_paths: Optional[List[str]],
        conversation_messages: Optional[list] = None,
    ) -> Optional[AgentAction]:
        """Call LLM to decide what to do next based on current state."""
        llm = get_llm()

        has_structured = "Yes" if structured_file_paths else "No"
        current_q = state.current_sub_question or state.original_query

        system_prompt = (
            "You are an advanced AI Agent tasked with answering complex user questions.\n"
            "You have access to the following tools:\n"
            "- hybrid_search: Searches text documents (PDFs, TXT, DOCX) for semantic meaning.\n"
            "- graph_search: Searches the knowledge graph for relationships and entity connections.\n"
            "- pandas_qa: Executes python code to answer numerical or exact analytical questions on structured data (CSV/Excel).\n"
            "- final_answer: Provide the final answer to the user when you have enough evidence.\n\n"
            f"Note: Structured data files available? {has_structured}\n\n"
            "## Self-Correction Rules\n"
            "- If the most recent observation has LOW relevance scores (below 0.3), "
            "the evidence may be poor. You SHOULD rewrite or rephrase the query and search again.\n"
            "- If the observation says 'No relevant documents found', try a broader or rephrased query.\n"
            "- If you see contradictory evidence, search for clarification before giving a final answer.\n"
            "- If you have gathered sufficient high-quality evidence, call 'final_answer'.\n"
            "- Do NOT repeat the exact same tool and query combination you already tried.\n\n"
            "Respond in JSON matching this schema:\n"
            '{"action": {"tool_name": "hybrid_search|graph_search|pandas_qa|final_answer", '
            '"query": "tool input", "reasoning": "why"}}'
        )

        # Build conversation history
        messages = [{"role": "system", "content": system_prompt}]
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Original Query: {state.original_query}\n"
                    f"Current Sub-Question: {current_q}"
                ),
            }
        )

        for idx, step in enumerate(state.steps):
            confidence_tag = f" [Confidence: {step.confidence:.3f}]"
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Tool Called: {step.tool}\nQuery: {step.query}{confidence_tag}",
                }
            )
            truncated_obs = _smart_truncate(step.observation, max_chars=2000)
            messages.append(
                {"role": "user", "content": f"Observation {idx + 1}:\n{truncated_obs}"}
            )

        messages.append(
            {"role": "user", "content": "What is the next action? Return JSON only."}
        )

        # Inject conversation context if available
        if conversation_messages and not state.steps:
            conv_context = "\n".join(
                f"{getattr(m, 'role', 'user').title()}: {getattr(m, 'content', str(m))}"
                for m in conversation_messages[-6:]
            )
            messages.insert(1, {
                "role": "user",
                "content": f"Prior conversation context (for follow-up understanding):\n{conv_context}",
            })

        try:
            response = await llm.generate(
                messages,
                temperature=0.0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            data = json.loads(response)
            self._token_usage += _estimate_tokens(response) + sum(
                _estimate_tokens(m.get("content", "")) for m in messages
            )
            return AgentAction(**data)

        except Exception as e:
            logger.error("Failed to plan next action: %s", e)
            return None

    # ── Tool Execution ──────────────────────────────────────

    async def _execute_tool(
        self,
        tool_name: str,
        query: str,
        document_ids: Optional[List[UUID]],
        structured_file_paths: Optional[List[str]],
        owner_key: Optional[str] = None,
    ) -> tuple[str, List[Dict[str, Any]], List[float]]:
        """Dispatch to the appropriate tool wrapper. Returns (observation, citations, scores)."""

        logger.info("Executing Tool: %s with query: '%s'", tool_name, query)

        if tool_name == "hybrid_search":
            return await self.hybrid_tool.execute(query, document_ids, owner_key=owner_key)

        elif tool_name == "graph_search":
            observation, scores = await self.graph_tool.execute(query)
            return observation, [], scores

        elif tool_name == "pandas_qa":
            observation = await self.pandas_tool.execute(query, structured_file_paths)
            return observation, [], []

        return f"Unknown tool: {tool_name}", [], []

    # ── Evidence Confidence ─────────────────────────────────

    @staticmethod
    def _compute_confidence(relevance_scores: List[float]) -> float:
        """
        Compute a confidence score (0-1) from per-chunk relevance scores.

        Uses a weighted average that gives more weight to the top-scoring chunk,
        with a penalty for very few results (single-chunk retrievals shouldn't
        score 1.0 without very high relevance).
        """
        if not relevance_scores:
            return 0.0

        sorted_scores = sorted(relevance_scores, reverse=True)
        # Weight: top score counts double
        weights = [2.0] + [1.0] * (len(sorted_scores) - 1)
        weighted_sum = sum(s * w for s, w in zip(sorted_scores, weights))
        total_weight = sum(weights)

        raw_confidence = min(weighted_sum / total_weight, 1.0)

        # Penalty for very few results: reduce confidence if only 1-2 chunks found
        count_penalty = min(len(relevance_scores) / 3.0, 1.0)
        return raw_confidence * count_penalty

    # ── Final Answer Generation ─────────────────────────────

    async def _generate_final_answer(self, state: AgentState) -> str:
        """Generate the final grounded answer from all accumulated observations and sub-answers."""
        llm = get_llm()

        system_prompt = (
            "You are an Advanced AI Answering Assistant known for providing **thorough, in-depth, and comprehensive** answers. "
            "Based on the gathered evidence, provide a detailed, well-structured, and expert-level answer to the original query.\n\n"
            "Rules:\n"
            "- **Be thorough**: Go beyond surface-level summaries. Explain concepts, provide context, give examples, and cover nuances. "
            "Your answer should satisfy a knowledgeable professional.\n"
            "- If the evidence contains contradictions, explain them clearly with specifics from each source.\n"
            "- If the gathered evidence only partially answers the query or lacks direct information, DO NOT give a blunt refusal. Instead, organize your response into:\n"
            "  1. **Document Context & Findings**: What relevant or related facts exist in the documents (citing source document names). Thoroughly extract all useful information.\n"
            "  2. **Answer from AI Knowledge**: A **detailed, comprehensive, expert-level** answer to the user's question using general AI knowledge "
            "(under callout: '> 💡 **General Knowledge Answer**: ...'). Include definitions, explanations, examples, comparisons, and practical insights.\n"
            "  3. **Recommended Plan & Next Steps**: Actionable next steps, suggestions on specific data or files to upload for grounded verification, or recommended follow-up questions.\n"
            "- Cite document evidence by referencing the source document names.\n"
            "- If sub-questions were answered, synthesize them into a cohesive, well-structured final answer that reads as a unified response.\n"
            "- Use rich formatting: headings, subheadings, bullet points, numbered lists, tables, and callouts.\n"
            f"- Overall evidence confidence: {state.overall_confidence:.2f}/1.00"
        )

        # Build evidence section
        evidence_parts = []
        for step in state.steps:
            evidence_parts.append(
                f"--- Evidence from {step.tool} (confidence: {step.confidence:.3f}) ---\n"
                f"{step.observation}"
            )
        evidence = "\n\n".join(evidence_parts)

        # Build sub-answers section if decomposed
        sub_answer_text = ""
        if len(state.sub_answers) > 1:
            sub_parts = []
            for sq, sa in state.sub_answers.items():
                sub_parts.append(f"Sub-Q: {sq}\nSub-A: {sa}")
            sub_answer_text = (
                "\n\nSub-Question Answers:\n" + "\n\n".join(sub_parts)
            )

        user_prompt = (
            f"Original Query: {state.original_query}\n\n"
            f"Accumulated Evidence:\n{evidence}"
            f"{sub_answer_text}"
        )

        try:
            response = await llm.generate(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=4096,
            )
            return response
        except Exception as e:
            logger.error("Failed to generate final answer: %s", e)
            return (
                "I analyzed the documents, but encountered difficulty formulating a complete final response. "
                "Please try asking your question again or rephrasing."
            )


# ── Utility Functions ───────────────────────────────────────


def _smart_truncate(text: str, max_chars: int = 2000) -> str:
    """
    Truncate text intelligently, preserving complete sentences and adding
    a clear marker instead of a bare '...' (fixes task 2.1).
    """
    if len(text) <= max_chars:
        return text

    # Try to cut at a sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind(". ")
    last_newline = truncated.rfind("\n")
    cut_point = max(last_period, last_newline)

    if cut_point > max_chars * 0.5:
        truncated = truncated[: cut_point + 1]
    # else keep the hard cut at max_chars

    remaining = len(text) - len(truncated)
    return f"{truncated}\n\n[TRUNCATED — {remaining} characters omitted]"
