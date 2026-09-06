"""
Prompt templates for the RAG answer generation pipeline.
Enforces grounding, citation, and evidence-based answering (PRD §17).
"""

# ── System Prompt ───────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are an Advanced RAG Assistant — an AI that answers questions using ONLY the provided source documents.

## Core Rules
1. **Evidence-Grounded**: Every factual claim in your answer MUST be supported by the provided evidence. Do NOT use outside knowledge.
2. **Citations Required**: Reference sources using [1], [2], etc. matching the source numbers provided.
3. **Honesty**: If the evidence is insufficient to answer the question, explicitly say: "The provided sources do not contain enough information to answer this question reliably."
4. **No Hallucination**: Never fabricate information, statistics, or claims not found in the sources.
5. **Structured Answers**: Use clear formatting — headings, bullet points, numbered lists — for readability.
6. **Direct & Concise**: Answer the question directly first, then provide supporting details.

## Answer Format
- Start with a direct answer to the question
- Support with evidence from the sources using [N] citations
- End with a "Sources" section listing the referenced documents
"""

# ── Context Formatting Template ─────────────────────────────

CONTEXT_TEMPLATE = """## Retrieved Sources

{context}

## User Question
{query}

## Instructions
Answer the question using ONLY the sources above. Cite each source using its [N] number.
If sources conflict, mention the conflict and explain which source you consider more reliable and why.
If evidence is insufficient, say so explicitly.
"""

# ── Conversation Context Template ───────────────────────────

CONVERSATION_CONTEXT_TEMPLATE = """## Prior Conversation Context
The following is the recent conversation history for context. Use it to understand follow-up questions and maintain coherence, but still ground all factual claims in the retrieved sources above.

{conversation_history}
"""


def format_context(chunks: list, query: str) -> str:
    """
    Format retrieved chunks into the context section of the prompt.

    Args:
        chunks: List of RetrievedChunk objects.
        query: The user's original query.

    Returns:
        Formatted context string ready for the LLM.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        source_info = []
        if chunk.source_filename:
            source_info.append(f"Source: {chunk.source_filename}")
        if chunk.page_number:
            source_info.append(f"Page: {chunk.page_number}")
        if chunk.section:
            source_info.append(f"Section: {chunk.section}")

        header = f"[{i}] " + " | ".join(source_info) if source_info else f"[{i}]"
        context_parts.append(f"{header}\n{chunk.content}\n")

    context_text = "\n---\n".join(context_parts)
    return CONTEXT_TEMPLATE.format(context=context_text, query=query)


def format_conversation_history(messages: list) -> str:
    """
    Format conversation messages into a prompt section.

    Args:
        messages: List of message objects with .role and .content attributes.

    Returns:
        Formatted conversation history string.
    """
    if not messages:
        return ""

    parts = []
    for msg in messages:
        role = getattr(msg, "role", "user")
        content = getattr(msg, "content", str(msg))
        label = "User" if role == "user" else "Assistant"
        parts.append(f"**{label}**: {content}")

    history_text = "\n\n".join(parts)
    return CONVERSATION_CONTEXT_TEMPLATE.format(conversation_history=history_text)


def build_messages(context_prompt: str, conversation_history: str = "") -> list:
    """
    Build the full message list for the LLM.

    Args:
        context_prompt: The formatted context with sources and query.
        conversation_history: Optional formatted prior conversation context.

    Returns:
        List of message dicts: [{role, content}, ...]
    """
    user_content = context_prompt
    if conversation_history:
        user_content = conversation_history + "\n\n" + context_prompt

    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
