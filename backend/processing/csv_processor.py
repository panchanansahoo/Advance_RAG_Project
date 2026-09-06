"""
Processor for structured tabular data (CSV, Excel).

Phase 3 component. Extracts tabular data as formatted text/markdown.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from backend.processing.base import BaseProcessor, ProcessedContent
from backend.schemas.common import ContentType

logger = logging.getLogger(__name__)


class CSVProcessor(BaseProcessor):
    """
    Extracts text from structured tabular data formats (CSV, XLSX).
    Converts tables to Markdown format for better chunking.
    """

    def supported_extensions(self) -> List[str]:
        return [".csv", ".xlsx", ".xls"]

    async def process(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[ProcessedContent]:
        """
        Process a CSV or Excel file.
        """
        metadata = metadata or {}
        contents: List[ProcessedContent] = []
        
        try:
            if file_path.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_path)
                file_type = "excel"
            else:
                # For CSV we need to handle potential encoding issues
                import chardet
                with open(file_path, 'rb') as f:
                    detector = chardet.detect(f.read(10000))
                encoding = detector['encoding'] or 'utf-8'
                df = pd.read_csv(file_path, encoding=encoding)
                file_type = "csv"
                
        except Exception as e:
            logger.error("Failed to parse tabular data: %s", e)
            raise ValueError(f"Invalid or unsupported tabular data: {e}")

        # Convert DataFrame to Markdown for LLM readability
        markdown_table = df.to_markdown(index=False)

        contents.append(
            ProcessedContent(
                text=markdown_table,
                content_type=ContentType.TABLE,
                page_number=1,
                metadata={
                    **metadata,
                    "source": "pandas",
                    "file_type": file_type,
                    "rows": len(df),
                    "columns": len(df.columns),
                },
            )
        )

        logger.info("Tabular data processed: %s → %d rows, %d columns", file_path, len(df), len(df.columns))
        return contents
