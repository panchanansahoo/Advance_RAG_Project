"""Tests for Verification Service."""

import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from backend.schemas.queries import Citation
from backend.verification.service import VerificationService


class TestVerificationService:
    """Tests for the VerificationService."""

    @patch("backend.verification.service.get_settings")
    @pytest.mark.asyncio
    async def test_verification_disabled(self, mock_settings):
        settings = mock_settings.return_value
        settings.verification_enabled = False

        verifier = VerificationService()
        result = await verifier.verify("query", "Answer", [])
        
        assert result.is_fully_supported is True
        assert result.revised_answer == "Answer"

    @patch("backend.verification.service.get_settings")
    @pytest.mark.asyncio
    async def test_verification_no_citations(self, mock_settings):
        settings = mock_settings.return_value
        settings.verification_enabled = True

        verifier = VerificationService()
        result = await verifier.verify("query", "Answer", [])
        
        assert result.is_fully_supported is True
        assert result.revised_answer == "Answer"

    @patch("backend.verification.service.get_llm")
    @patch("backend.verification.service.get_settings")
    @pytest.mark.asyncio
    async def test_verify_fully_supported(self, mock_settings, mock_get_llm):
        settings = mock_settings.return_value
        settings.verification_enabled = True

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = '''
        {
            "is_fully_supported": true,
            "claims": [{"text": "Sky is blue", "is_supported": true, "reasoning": "stated in doc"}],
            "contradictions": []
        }
        '''
        mock_get_llm.return_value = mock_llm

        citation = Citation(
            document_id=uuid4(),
            document_name="test.txt",
            chunk_id=uuid4(),
            content_snippet="The sky is blue."
        )

        verifier = VerificationService()
        result = await verifier.verify("What color is the sky?", "The sky is blue.", [citation])
        
        assert result.is_fully_supported is True
        assert "WARNING" not in result.revised_answer
        assert result.revised_answer == "The sky is blue."

    @patch("backend.verification.service.get_llm")
    @patch("backend.verification.service.get_settings")
    @pytest.mark.asyncio
    async def test_verify_unsupported_claim(self, mock_settings, mock_get_llm):
        settings = mock_settings.return_value
        settings.verification_enabled = True

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = '''
        {
            "is_fully_supported": false,
            "claims": [{"text": "Sky is green", "is_supported": false, "reasoning": "not in doc"}],
            "contradictions": []
        }
        '''
        mock_get_llm.return_value = mock_llm

        citation = Citation(
            document_id=uuid4(),
            document_name="test.txt",
            chunk_id=uuid4(),
            content_snippet="The sky is blue."
        )

        verifier = VerificationService()
        result = await verifier.verify("What color is the sky?", "The sky is green.", [citation])
        
        assert result.is_fully_supported is False
        assert "WARNING" in result.revised_answer
        assert "Unsupported Claims Detected" in result.revised_answer
        assert "Sky is green" in result.revised_answer

    @patch("backend.verification.service.get_llm")
    @patch("backend.verification.service.get_settings")
    @pytest.mark.asyncio
    async def test_verify_contradiction(self, mock_settings, mock_get_llm):
        settings = mock_settings.return_value
        settings.verification_enabled = True

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = '''
        {
            "is_fully_supported": false,
            "claims": [],
            "contradictions": [{"description": "Color differs", "source_a": "doc1.txt", "source_b": "doc2.txt"}]
        }
        '''
        mock_get_llm.return_value = mock_llm

        citation1 = Citation(
            document_id=uuid4(),
            document_name="doc1.txt",
            chunk_id=uuid4(),
            content_snippet="The sky is blue."
        )
        citation2 = Citation(
            document_id=uuid4(),
            document_name="doc2.txt",
            chunk_id=uuid4(),
            content_snippet="The sky is red."
        )

        verifier = VerificationService()
        result = await verifier.verify("What color is the sky?", "The sky is blue and red.", [citation1, citation2])
        
        assert result.is_fully_supported is False
        assert "IMPORTANT" in result.revised_answer
        assert "Conflicting Evidence Detected" in result.revised_answer
        assert "Color differs" in result.revised_answer
