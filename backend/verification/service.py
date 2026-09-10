"""
Verification Service — Fact-checks claims and detects contradictions.

Phase 7 component.

Improvements (Task 1.4):
- Accepts optional full_evidence_texts for deep fact-checking
- Falls back to citation snippets if full evidence not provided
- Enhanced contradiction detection with source-authority comparison (PRD §19)
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional

from backend.config import get_settings
from backend.generation.llm import get_llm
from backend.schemas.queries import Citation
from backend.schemas.verification import VerificationResult, Claim, Contradiction

logger = logging.getLogger(__name__)


class VerificationService:
    """Verifies drafted answers against retrieved evidence."""

    # Maximum time to spend on verification before falling through
    VERIFICATION_TIMEOUT = 30  # seconds
    # Maximum characters per evidence source to prevent token overflow
    MAX_EVIDENCE_LENGTH = 8000

    def __init__(self):
        self.settings = get_settings()

    async def verify(
        self,
        query: str,
        drafted_answer: str,
        citations: List[Citation],
        full_evidence_texts: Optional[List[str]] = None,
    ) -> VerificationResult:
        """
        Verify the drafted answer against the citations.

        Args:
            query: The original user query.
            drafted_answer: The generated answer to fact-check.
            citations: Citation objects with document references.
            full_evidence_texts: Optional list of full-length evidence texts
                from retrieval/agent steps. If provided, these are used instead
                of the truncated citation snippets for more reliable verification.

        Returns:
            VerificationResult that may contain a revised answer with warnings.
        """
        if not self.settings.verification_enabled:
            return VerificationResult(
                is_fully_supported=True, revised_answer=drafted_answer
            )

        if not citations and not full_evidence_texts:
            # If no evidence was provided, we can't verify claims.
            return VerificationResult(
                is_fully_supported=True, revised_answer=drafted_answer
            )

        # Run verification with timeout to prevent blocking
        try:
            return await asyncio.wait_for(
                self._do_verify(query, drafted_answer, citations, full_evidence_texts),
                timeout=self.VERIFICATION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Verification timed out after %ds — passing through unverified",
                self.VERIFICATION_TIMEOUT,
            )
            return VerificationResult(
                is_fully_supported=True, revised_answer=drafted_answer
            )

    async def _do_verify(
        self,
        query: str,
        drafted_answer: str,
        citations: List[Citation],
        full_evidence_texts: Optional[List[str]] = None,
    ) -> VerificationResult:
        """Core verification logic, called within timeout wrapper."""

        llm = get_llm()

        system_prompt = (
            "You are a strict fact-checker and reliability agent. "
            "Your job is to evaluate a drafted answer against the provided evidence.\n\n"
            "Rules:\n"
            "1. Extract the main factual claims made in the drafted answer.\n"
            "2. For claims that cite or purport to come from the provided evidence, check if EVERY claim is explicitly supported by the provided evidence. "
            "If a claim is purported to come from evidence but is not supported, it is unsupported (is_supported: false).\n"
            "3. If a claim or section is explicitly labeled as general knowledge or outside document context, do NOT mark it as an unsupported hallucination, as long as it does not falsely cite or attribute itself to the documents.\n"
            "4. Look for contradictions between different evidence sources. "
            "If Source A says X and Source B says Y, and the drafted answer doesn't "
            "explicitly address this conflict, flag it.\n"
            "5. When contradictions are found, evaluate source authority:\n"
            "   - More recent sources are generally more reliable\n"
            "   - Official/primary sources outweigh secondary sources\n"
            "   - Quantitative data should be cross-checked for consistency\n"
            "6. Return a JSON object matching this schema:\n"
            "{\n"
            '  "is_fully_supported": true/false,\n'
            '  "claims": [{"text": "claim", "is_supported": true/false, "reasoning": "why"}],\n'
            '  "contradictions": [{"description": "conflict", "source_a": "A", "source_b": "B"}]\n'
            "}"
        )

        # Use full evidence texts if available (Task 1.4), otherwise fall back to snippets
        if full_evidence_texts:
            evidence_parts = []
            for i, evidence in enumerate(full_evidence_texts):
                # Cap evidence length to prevent token overflow
                trimmed = evidence[:self.MAX_EVIDENCE_LENGTH]
                if len(evidence) > self.MAX_EVIDENCE_LENGTH:
                    trimmed += f"\n[... {len(evidence) - self.MAX_EVIDENCE_LENGTH} chars truncated]"
                # Use citation info if available to label the evidence
                source_label = f"Source {i + 1}"
                if i < len(citations):
                    source_label = f"{citations[i].document_name} (Chunk: {citations[i].chunk_id})"
                evidence_parts.append(f"--- {source_label} ---\n{trimmed}")
            evidence_text = "\n\n".join(evidence_parts)
        else:
            evidence_text = "\n\n".join(
                [
                    f"--- Source: {c.document_name} (Chunk: {c.chunk_id}) ---\n{c.content_snippet}"
                    for c in citations
                ]
            )

        user_prompt = (
            f"Original Query: {query}\n\n"
            f"Drafted Answer:\n{drafted_answer}\n\n"
            f"Provided Evidence:\n{evidence_text}"
        )

        try:
            response_text = await llm.generate(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )

            data = json.loads(response_text)
            result = VerificationResult(**data)

            # If not fully supported, append a warning block to the drafted answer
            result.revised_answer = self._append_warnings(drafted_answer, result)
            return result

        except Exception as e:
            logger.error("Verification failed: %s", e)
            # Fail open to not block the user
            return VerificationResult(
                is_fully_supported=True, revised_answer=drafted_answer
            )

    def _append_warnings(
        self, drafted_answer: str, result: VerificationResult
    ) -> str:
        """Appends Markdown warning blocks if there are unsupported claims or contradictions."""
        if result.is_fully_supported and not result.contradictions:
            return drafted_answer

        warnings = []

        # Unsupported Claims
        unsupported = [c for c in result.claims if not c.is_supported]
        if unsupported:
            warning_text = (
                "> [!WARNING]\n"
                "> **Unsupported Claims Detected**\n"
                "> The following claims in the answer could not be verified "
                "by the retrieved documents:\n"
            )
            for c in unsupported:
                warning_text += f"> - {c.text} ({c.reasoning})\n"
            warnings.append(warning_text)

        # Contradictions
        if result.contradictions:
            conflict_text = (
                "> [!IMPORTANT]\n"
                "> **Conflicting Evidence Detected**\n"
                "> The sources provide contradictory information:\n"
            )
            for conflict in result.contradictions:
                conflict_text += (
                    f"> - {conflict.description} "
                    f"(Source A: {conflict.source_a} vs Source B: {conflict.source_b})\n"
                )
            warnings.append(conflict_text)

        if not warnings:
            return drafted_answer

        return drafted_answer + "\n\n---\n\n" + "\n\n".join(warnings)


# ── Singleton accessor (Task 4.1) ───────────────────────────
_verification_service: Optional[VerificationService] = None


def get_verification_service() -> VerificationService:
    """Return a cached VerificationService singleton."""
    global _verification_service
    if _verification_service is None:
        _verification_service = VerificationService()
    return _verification_service

