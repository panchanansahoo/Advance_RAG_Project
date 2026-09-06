"""
Text and Markdown file processors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chardet

from backend.processing.base import BaseProcessor, ProcessedContent
from backend.schemas.common import ContentType

logger = logging.getLogger(__name__)


class TextProcessor(BaseProcessor):
    """Process plain text files."""

    def supported_extensions(self) -> List[str]:
        return [".txt"]

    async def process(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[ProcessedContent]:
        metadata = metadata or {}
        text = self._read_file(file_path)

        if not text.strip():
            logger.warning("Empty text file: %s", file_path)
            return []

        return [
            ProcessedContent(
                text=text.strip(),
                content_type=ContentType.TEXT,
                metadata={**metadata, "source": "text_parser"},
            )
        ]

    @staticmethod
    def _read_file(file_path: str) -> str:
        """Read a text file with encoding detection."""
        raw = Path(file_path).read_bytes()
        detected = chardet.detect(raw)
        encoding = detected.get("encoding", "utf-8") or "utf-8"
        return raw.decode(encoding, errors="replace")


class MarkdownProcessor(BaseProcessor):
    """Process Markdown files — preserves structure for smarter chunking."""

    def supported_extensions(self) -> List[str]:
        return [".md"]

    async def process(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[ProcessedContent]:
        metadata = metadata or {}
        raw = Path(file_path).read_bytes()
        detected = chardet.detect(raw)
        encoding = detected.get("encoding", "utf-8") or "utf-8"
        text = raw.decode(encoding, errors="replace")

        if not text.strip():
            return []

        # Split by top-level headings for structure awareness
        sections = self._split_by_headings(text)
        contents: List[ProcessedContent] = []

        for section_title, section_text in sections:
            if section_text.strip():
                contents.append(
                    ProcessedContent(
                        text=section_text.strip(),
                        content_type=ContentType.TEXT,
                        section=section_title,
                        metadata={**metadata, "source": "markdown_parser"},
                    )
                )

        return contents

    @staticmethod
    def _split_by_headings(text: str) -> List[tuple]:
        """Split markdown text by headings, returning (heading, body) tuples."""
        import re

        sections = []
        current_heading = None
        current_lines = []

        for line in text.split("\n"):
            heading_match = re.match(r"^(#{1,3})\s+(.+)$", line)
            if heading_match:
                # Save previous section
                if current_lines:
                    sections.append(
                        (current_heading, "\n".join(current_lines))
                    )
                current_heading = heading_match.group(2).strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        # Save last section
        if current_lines:
            sections.append((current_heading, "\n".join(current_lines)))

        return sections
