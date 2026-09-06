"""Tests for Vision Language Models."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image

from backend.processing.vlm.gemini_vlm import GeminiVLM
from backend.processing.vlm.openai_vlm import OpenAIVLM


class TestGeminiVLM:
    """Tests for GeminiVLM."""

    @patch("backend.processing.vlm.gemini_vlm.genai")
    @pytest.mark.asyncio
    async def test_analyze_image_success(self, mock_genai):
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "A beautiful sunset over the mountains."
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        vlm = GeminiVLM()
        image = Image.new("RGB", (100, 100))
        
        result = await vlm.analyze_image(image, prompt="What is this?")
        
        assert result == "A beautiful sunset over the mountains."
        mock_model.generate_content.assert_called_once()
        args = mock_model.generate_content.call_args[0][0]
        assert "What is this?" in args
        assert image in args

    @patch("backend.processing.vlm.gemini_vlm.genai")
    @pytest.mark.asyncio
    async def test_analyze_image_failure(self, mock_genai):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_genai.GenerativeModel.return_value = mock_model

        vlm = GeminiVLM()
        image = Image.new("RGB", (100, 100))
        
        result = await vlm.analyze_image(image)
        
        assert result == ""


class TestOpenAIVLM:
    """Tests for OpenAIVLM."""

    @patch("backend.processing.vlm.openai_vlm.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_analyze_image_success(self, mock_openai_cls):
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="A bar chart showing sales."))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        vlm = OpenAIVLM()
        image = Image.new("RGB", (100, 100))
        
        result = await vlm.analyze_image(image, prompt="What is this?")
        
        assert result == "A bar chart showing sales."
        mock_client.chat.completions.create.assert_called_once()

    @patch("backend.processing.vlm.openai_vlm.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_analyze_image_failure(self, mock_openai_cls):
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        vlm = OpenAIVLM()
        image = Image.new("RGB", (100, 100))
        
        result = await vlm.analyze_image(image)
        
        assert result == ""
