"""
Conversation Memory Manager — implements PRD §20 memory strategies.

Provides three memory tiers:
1. Short-term: Last N raw messages (existing behavior)
2. Summary: LLM-generated summaries of older messages
3. Relevant: Combination of summary + recent messages for the LLM prompt

Task 3.3 component.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from backend.config import get_settings
from backend.generation.llm import get_llm

logger = logging.getLogger(__name__)

# Threshold: summarize when conversation exceeds this many messages
SUMMARY_THRESHOLD = 12
RECENT_MESSAGES_COUNT = 6  # Keep last N messages as raw context


class ConversationMemoryManager:
    """
    Manages conversation context with short-term memory and
    progressive summarization for long conversations (PRD §20).
    """

    def __init__(self):
        self.settings = get_settings()

    async def build_context(
        self,
        messages: list,
        existing_summary: Optional[str] = None,
    ) -> tuple[str, Optional[str]]:
        """
        Build conversation context from messages.

        For short conversations: returns raw messages.
        For long conversations: summarizes older messages, keeps recent ones.

        Args:
            messages: Full list of conversation messages (oldest first).
            existing_summary: Previously generated summary (if any).

        Returns:
            Tuple of (context_string, updated_summary).
            The updated_summary should be stored for future calls.
        """
        if not messages:
            return "", None

        total = len(messages)

        # Short conversation — just return recent messages
        if total <= SUMMARY_THRESHOLD:
            return self._format_messages(messages), existing_summary

        # Long conversation — summarize older messages, keep recent ones
        recent = messages[-RECENT_MESSAGES_COUNT:]
        older = messages[:-RECENT_MESSAGES_COUNT]

        # Generate or update summary
        updated_summary = await self._generate_summary(older, existing_summary)

        # Build combined context
        context_parts = []
        if updated_summary:
            context_parts.append(
                f"**Conversation Summary** (covers earlier messages):\n{updated_summary}"
            )
        context_parts.append(
            "**Recent Messages:**\n" + self._format_messages(recent)
        )

        return "\n\n---\n\n".join(context_parts), updated_summary

    async def _generate_summary(
        self,
        older_messages: list,
        existing_summary: Optional[str] = None,
    ) -> str:
        """
        Generate a progressive summary of older conversation messages.

        If an existing summary is provided, it's updated with the new messages
        rather than regenerated from scratch (progressive summarization).
        """
        llm = get_llm()

        messages_text = self._format_messages(older_messages)

        if existing_summary:
            system_prompt = (
                "You are a conversation summarizer. You have a previous summary of an "
                "ongoing conversation and new messages to incorporate. "
                "Update the summary to include the new information while keeping it concise.\n\n"
                "Rules:\n"
                "1. Preserve key facts, decisions, and context from the previous summary\n"
                "2. Add new important information from the recent messages\n"
                "3. Keep the summary under 200 words\n"
                "4. Focus on information that would be useful for answering future questions\n"
                "5. Return ONLY the updated summary, nothing else"
            )
            user_prompt = (
                f"Previous Summary:\n{existing_summary}\n\n"
                f"New Messages to Incorporate:\n{messages_text}"
            )
        else:
            system_prompt = (
                "You are a conversation summarizer. Summarize the following conversation "
                "into a concise summary that captures:\n"
                "1. The main topics discussed\n"
                "2. Key questions asked and answers given\n"
                "3. Any important facts, decisions, or context\n"
                "4. Keep the summary under 200 words\n"
                "5. Return ONLY the summary, nothing else"
            )
            user_prompt = f"Conversation to summarize:\n{messages_text}"

        try:
            summary = await llm.generate(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            logger.info(
                "Conversation summary generated (%d chars from %d messages)",
                len(summary),
                len(older_messages),
            )
            return summary.strip()
        except Exception as e:
            logger.warning("Failed to generate conversation summary: %s", e)
            return existing_summary or ""

    @staticmethod
    def _format_messages(messages: list) -> str:
        """Format a list of messages into a readable string."""
        parts = []
        for msg in messages:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", str(msg))
            label = "User" if role == "user" else "Assistant"
            parts.append(f"**{label}**: {content}")
        return "\n\n".join(parts)
