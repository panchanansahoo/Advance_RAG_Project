"""
Structure-aware chunking — respects document structure boundaries.

Handles:
- Markdown / text headings (section boundaries)
- Table blocks (kept as atomic units — table-aware)
- Bullet / numbered lists (grouped together)
- Respects max chunk size while preferring structural breaks
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from backend.processing.chunking.base import BaseChunker, ChunkData
from backend.schemas.common import ContentType

logger = logging.getLogger(__name__)

# ── Regex patterns for structural elements ──────────────────

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
TABLE_ROW_RE = re.compile(r"^\|(.+)\|$")
TABLE_SEPARATOR_RE = re.compile(r"^\|[\s\-:]+\|$")
LIST_ITEM_RE = re.compile(r"^(\s*)([-*•]|\d+[.)]) ")


class StructureAwareChunker(BaseChunker):
    """
    Split text into chunks that respect document structure.

    Priority order for boundaries:
    1. Heading boundaries (never split across headings)
    2. Table boundaries (tables are kept atomic)
    3. List boundaries
    4. Paragraph boundaries
    5. Sentence boundaries (last resort within paragraphs)
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def strategy_name(self) -> str:
        return "structure_aware"

    def chunk(self, text: str, **kwargs) -> List[ChunkData]:
        if not text or not text.strip():
            return []

        page_number = kwargs.get("page_number")
        section = kwargs.get("section")
        content_type = kwargs.get("content_type", ContentType.TEXT)
        start_index = kwargs.get("start_index", 0)

        # Step 1: Parse into structural blocks
        blocks = self._parse_into_blocks(text)

        # Step 2: Group blocks into chunks respecting size limits
        chunk_groups = self._group_blocks(blocks)

        # Step 3: Build ChunkData objects
        chunks: List[ChunkData] = []
        idx = start_index
        for group in chunk_groups:
            chunk_text = "\n\n".join(b["text"] for b in group).strip()
            if not chunk_text:
                continue

            # Determine content type from blocks
            block_types = {b["type"] for b in group}
            if "table" in block_types:
                chunk_content_type = ContentType.TABLE
            else:
                chunk_content_type = content_type

            # Get section from first heading block
            chunk_section = section
            for b in group:
                if b["type"] == "heading":
                    chunk_section = b.get("heading_text", section)
                    break

            chunks.append(
                ChunkData(
                    content=chunk_text,
                    chunk_index=idx,
                    content_type=chunk_content_type,
                    page_number=page_number,
                    section=chunk_section,
                    token_count=len(chunk_text.split()),
                    metadata={
                        "block_types": list(block_types),
                        "heading_level": next(
                            (b.get("heading_level", 0) for b in group if b["type"] == "heading"), 0
                        ),
                    },
                )
            )
            idx += 1

        return chunks

    def _parse_into_blocks(self, text: str) -> List[dict]:
        """
        Parse text into structural blocks.
        Each block has: type, text, and optional metadata.
        """
        blocks: List[dict] = []
        lines = text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # ── Heading ─────────────────────────────────
            heading_match = HEADING_RE.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                blocks.append({
                    "type": "heading",
                    "text": line,
                    "heading_text": heading_text,
                    "heading_level": level,
                })
                i += 1
                continue

            # ── Table ───────────────────────────────────
            if TABLE_ROW_RE.match(line.strip()) or TABLE_SEPARATOR_RE.match(line.strip()):
                table_lines = []
                while i < len(lines) and (
                    TABLE_ROW_RE.match(lines[i].strip())
                    or TABLE_SEPARATOR_RE.match(lines[i].strip())
                    or lines[i].strip().startswith("|")
                ):
                    table_lines.append(lines[i])
                    i += 1
                if table_lines:
                    blocks.append({
                        "type": "table",
                        "text": "\n".join(table_lines),
                    })
                continue

            # ── List ────────────────────────────────────
            if LIST_ITEM_RE.match(line):
                list_lines = []
                while i < len(lines) and (
                    LIST_ITEM_RE.match(lines[i]) or (lines[i].startswith("  ") and list_lines)
                ):
                    list_lines.append(lines[i])
                    i += 1
                if list_lines:
                    blocks.append({
                        "type": "list",
                        "text": "\n".join(list_lines),
                    })
                continue

            # ── Paragraph ───────────────────────────────
            if line.strip():
                para_lines = []
                while i < len(lines) and lines[i].strip():
                    # Stop if we hit a heading, table, or list
                    if HEADING_RE.match(lines[i]):
                        break
                    if TABLE_ROW_RE.match(lines[i].strip()):
                        break
                    if LIST_ITEM_RE.match(lines[i]):
                        break
                    para_lines.append(lines[i])
                    i += 1
                if para_lines:
                    blocks.append({
                        "type": "paragraph",
                        "text": "\n".join(para_lines),
                    })
                continue

            # Skip blank lines
            i += 1

        return blocks

    def _group_blocks(self, blocks: List[dict]) -> List[List[dict]]:
        """
        Group blocks into chunks respecting size limits and structural boundaries.

        Rules:
        - Headings always start a new group (unless the group is empty)
        - Tables are kept atomic (if a table exceeds chunk_size, it's its own chunk)
        - Other blocks are accumulated until chunk_size is reached
        """
        if not blocks:
            return []

        groups: List[List[dict]] = []
        current_group: List[dict] = []
        current_length = 0

        for block in blocks:
            block_length = len(block["text"])

            # Headings of level 1-2 force a new chunk
            if (
                block["type"] == "heading"
                and block.get("heading_level", 99) <= 2
                and current_group
            ):
                groups.append(current_group)
                current_group = [block]
                current_length = block_length
                continue

            # If adding this block would exceed chunk size, flush
            if current_length + block_length > self.chunk_size and current_group:
                groups.append(current_group)
                current_group = []
                current_length = 0

            current_group.append(block)
            current_length += block_length

        # Flush remaining
        if current_group:
            groups.append(current_group)

        return groups
