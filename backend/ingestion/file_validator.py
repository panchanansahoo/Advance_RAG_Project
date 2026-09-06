"""
File validation for uploads — type, size, and extension checks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from backend.config import get_settings

logger = logging.getLogger(__name__)


class FileValidationError(Exception):
    """Raised when an uploaded file fails validation."""
    pass


class FileValidator:
    """Validates uploaded files against configured constraints."""

    def __init__(self):
        self.settings = get_settings()

    def validate(
        self,
        filename: str,
        file_size: int,
    ) -> str:
        """
        Validate a file upload.

        Args:
            filename: Original filename.
            file_size: File size in bytes.

        Returns:
            The file extension (lowercase, with dot).

        Raises:
            FileValidationError: If validation fails.
        """
        # Check filename
        if not filename or not filename.strip():
            raise FileValidationError("Filename is required")

        # Check extension
        ext = Path(filename).suffix.lower()
        if ext not in self.settings.allowed_extensions:
            raise FileValidationError(
                f"File type '{ext}' is not supported. "
                f"Allowed: {', '.join(self.settings.allowed_extensions)}"
            )

        # Check file size
        if file_size > self.settings.max_file_size_bytes:
            max_mb = self.settings.max_file_size_mb
            actual_mb = round(file_size / (1024 * 1024), 1)
            raise FileValidationError(
                f"File size ({actual_mb} MB) exceeds the maximum allowed size ({max_mb} MB)"
            )

        if file_size == 0:
            raise FileValidationError("File is empty")

        return ext

    def get_document_type(self, extension: str) -> str:
        """Map a file extension to a DocumentType value."""
        from backend.schemas.common import DocumentType

        mapping = {
            ".pdf": DocumentType.PDF,
            ".txt": DocumentType.TXT,
            ".md": DocumentType.MARKDOWN,
            ".csv": DocumentType.CSV,
            ".xlsx": DocumentType.XLSX,
            ".docx": DocumentType.DOCX,
            ".png": DocumentType.IMAGE,
            ".jpg": DocumentType.IMAGE,
            ".jpeg": DocumentType.IMAGE,
            ".html": DocumentType.HTML,
        }
        return mapping.get(extension, DocumentType.TXT)
