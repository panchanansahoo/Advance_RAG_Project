"""Tests for file validation and ingestion."""

import pytest
from backend.ingestion.file_validator import FileValidator, FileValidationError
from unittest.mock import patch


class TestFileValidator:
    """Tests for FileValidator."""

    @pytest.fixture
    def validator(self):
        with patch.dict("os.environ", {
            "ALLOWED_EXTENSIONS": '[".pdf", ".txt", ".md", ".csv"]',
            "MAX_FILE_SIZE_MB": "10",
            "CORS_ORIGINS": '["http://localhost:3000"]',
        }, clear=False):
            from backend.config.settings import Settings
            with patch("backend.ingestion.file_validator.get_settings") as mock:
                settings = Settings()
                mock.return_value = settings
                yield FileValidator()

    def test_valid_pdf(self, validator):
        ext = validator.validate("document.pdf", 1024 * 1024)
        assert ext == ".pdf"

    def test_valid_txt(self, validator):
        ext = validator.validate("notes.txt", 1024)
        assert ext == ".txt"

    def test_invalid_extension(self, validator):
        with pytest.raises(FileValidationError, match="not supported"):
            validator.validate("script.exe", 1024)

    def test_empty_filename(self, validator):
        with pytest.raises(FileValidationError, match="required"):
            validator.validate("", 1024)

    def test_file_too_large(self, validator):
        with pytest.raises(FileValidationError, match="exceeds"):
            validator.validate("big.pdf", 11 * 1024 * 1024)

    def test_empty_file(self, validator):
        with pytest.raises(FileValidationError, match="empty"):
            validator.validate("empty.pdf", 0)

    def test_case_insensitive_extension(self, validator):
        ext = validator.validate("Document.PDF", 1024)
        assert ext == ".pdf"
