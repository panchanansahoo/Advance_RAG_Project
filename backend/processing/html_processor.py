"""
Processor for HTML files.

Extracts text, headings, and tables from HTML documents using BeautifulSoup.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.processing.base import BaseProcessor, ProcessedContent
from backend.schemas.common import ContentType

logger = logging.getLogger(__name__)


class HTMLProcessor(BaseProcessor):
    """
    Extracts readable text content from HTML files.
    Strips scripts, styles, and navigation elements to focus on main content.
    """

    def supported_extensions(self) -> List[str]:
        return [".html", ".htm"]

    async def process(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[ProcessedContent]:
        metadata = metadata or {}
        contents: List[ProcessedContent] = []

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("beautifulsoup4 is not installed. Run: pip install beautifulsoup4")
            raise ValueError("beautifulsoup4 is required for HTML processing")

        try:
            import chardet

            raw = open(file_path, "rb").read()
            detected = chardet.detect(raw)
            encoding = detected.get("encoding", "utf-8") or "utf-8"
            html_text = raw.decode(encoding, errors="replace")
        except Exception as e:
            logger.error("Failed to read HTML file: %s", e)
            raise ValueError(f"Invalid HTML file: {e}")

        soup = BeautifulSoup(html_text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
            tag.decompose()

        # Extract title
        title = soup.title.string.strip() if soup.title and soup.title.string else None

        # Extract tables separately
        tables = soup.find_all("table")
        for table_idx, table in enumerate(tables):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if any(cells):
                    rows.append(" | ".join(cells))

            if rows:
                table_text = "\n".join(rows)
                contents.append(
                    ProcessedContent(
                        text=table_text,
                        content_type=ContentType.TABLE,
                        page_number=1,
                        section=title,
                        metadata={
                            **metadata,
                            "source": "html_parser",
                            "table_index": table_idx,
                        },
                    )
                )
            # Remove table from soup so it doesn't appear in main text
            table.decompose()

        # Extract main text content
        text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive blank lines
        import re
        text = re.sub(r"\n{3,}", "\n\n", text)

        if text.strip():
            contents.append(
                ProcessedContent(
                    text=text.strip(),
                    content_type=ContentType.TEXT,
                    page_number=1,
                    section=title,
                    metadata={
                        **metadata,
                        "source": "html_parser",
                        "html_title": title,
                    },
                )
            )

        logger.info("HTML processed: %s → %d content units", file_path, len(contents))
        return contents
