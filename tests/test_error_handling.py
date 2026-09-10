"""Unit tests for smart error sanitization and user-friendly formatting."""

import pytest
from backend.utils.error_sanitizer import (
    format_validation_error,
    sanitize_error_message,
    clean_response_answer,
)


class TestValidationErrors:
    """Tests for format_validation_error."""

    def test_missing_query(self):
        errors = [
            {"loc": ["body", "query"], "msg": "Field required", "type": "missing"}
        ]
        msg = format_validation_error(errors)
        assert "question" in msg.lower() or "prompt" in msg.lower()
        assert "Field required" not in msg
        assert "loc" not in msg

    def test_missing_file(self):
        errors = [
            {"loc": ["body", "file"], "msg": "Field required", "type": "missing"}
        ]
        msg = format_validation_error(errors)
        assert "document" in msg.lower() or "file" in msg.lower()
        assert "Field required" not in msg

    def test_invalid_uuid(self):
        errors = [
            {
                "loc": ["path", "conversation_id"],
                "msg": "Input should be a valid UUID",
                "type": "uuid_parsing",
            }
        ]
        msg = format_validation_error(errors)
        assert "identifier" in msg.lower() or "link" in msg.lower()
        assert "uuid_parsing" not in msg

    def test_empty_or_invalid_list(self):
        assert "verify your input" in format_validation_error([]).lower()
        assert "verify your input" in format_validation_error(None).lower()


class TestSanitizeErrorMessage:
    """Tests for sanitize_error_message."""

    def test_unsupported_file_extension(self):
        raw = "FileValidationError: File type '.exe' is not supported. Allowed: .pdf, .txt, .md, .csv"
        result = sanitize_error_message(raw)
        assert "not supported" in result.lower()
        assert "pdf" in result.lower()
        assert "FileValidationError" not in result

    def test_file_size_exceeded(self):
        raw = "File size (65.2 MB) exceeds the maximum allowed size (50 MB)"
        result = sanitize_error_message(raw)
        assert "exceeds" in result.lower()
        assert "65.2 mb" in result.lower()
        assert "50 mb" in result.lower()

    def test_empty_file(self):
        raw = "FileValidationError: File is empty"
        result = sanitize_error_message(raw)
        assert "empty" in result.lower()
        assert "FileValidationError" not in result

    def test_network_and_connection_errors(self):
        assert "momentarily unreachable" in sanitize_error_message("TypeError: Failed to fetch")
        assert "momentarily unreachable" in sanitize_error_message("ConnectionRefusedError: [Errno 111] Connection refused")
        assert "momentarily unreachable" in sanitize_error_message("ECONNREFUSED 127.0.0.1:8000")

    def test_rate_limiting_and_quota(self):
        assert "high demand" in sanitize_error_message("RateLimitError: 429 You exceeded your current quota").lower()
        assert "high demand" in sanitize_error_message("ResourceExhausted: quota exceeded").lower()

    def test_ai_auth_error(self):
        raw = "AuthenticationError: 401 Incorrect API key provided"
        result = sanitize_error_message(raw)
        assert "temporarily unavailable" in result.lower()
        assert "AuthenticationError" not in result
        assert "401" not in result

    def test_ai_model_busy(self):
        raw = "HTTP 503 The model is overloaded. Please try again later."
        result = sanitize_error_message(raw)
        assert "busy" in result.lower()
        assert "503" not in result

    def test_timeout_error(self):
        raw = "asyncio.TimeoutError: Task took longer than 30s"
        result = sanitize_error_message(raw)
        assert "longer than expected" in result.lower()
        assert "asyncio" not in result

    def test_resource_not_found(self):
        assert "could not be found" in sanitize_error_message("Document not found").lower()
        assert "could not be found" in sanitize_error_message("Conversation not found").lower()

    def test_pandas_security_block(self):
        raw = "Execution Error: Import of 'os' is not allowed for security reasons."
        result = sanitize_error_message(raw)
        assert "security reasons" in result.lower()
        assert "Execution Error" not in result
        assert "Import of" not in result

    def test_raw_python_traceback(self):
        # Tabular calculation traceback with KeyError
        tabular_trace = (
            "Traceback (most recent call last):\n"
            '  File "service.py", line 42, in process\n'
            "KeyError: 'Sales'\n"
        )
        result1 = sanitize_error_message(tabular_trace)
        assert "Traceback" not in result1
        assert "KeyError" not in result1
        assert "line 42" not in result1
        assert "spreadsheet" in result1.lower() or "calculation" in result1.lower()

        # Generic unexpected traceback with RuntimeError
        generic_trace = (
            "Traceback (most recent call last):\n"
            '  File "service.py", line 99, in unknown_step\n'
            "RuntimeError: invalid state encountered\n"
        )
        result2 = sanitize_error_message(generic_trace)
        assert "Traceback" not in result2
        assert "RuntimeError" not in result2
        assert "line 99" not in result2
        assert "temporary issue" in result2.lower()

    def test_database_error(self):
        raw = "sqlite3.OperationalError: no such table: document_chunks"
        result = sanitize_error_message(raw)
        assert "sqlite3" not in result
        assert "OperationalError" not in result
        assert "no such table" not in result
        assert "temporary issue" in result.lower()

    def test_already_friendly_string(self):
        friendly = "The revenue grew by 15% year over year."
        assert sanitize_error_message(friendly) == friendly


class TestCleanResponseAnswer:
    """Tests for clean_response_answer."""

    def test_pandas_execution_error_cleaned(self):
        raw = "Execution Error: KeyError: 'Revenue'"
        cleaned = clean_response_answer(raw)
        assert "Execution Error" not in cleaned
        assert "KeyError" not in cleaned
        assert "column names" in cleaned.lower()

    def test_syntax_error_in_code_cleaned(self):
        raw = "Syntax Error in generated code: invalid syntax (<string>, line 1)"
        cleaned = clean_response_answer(raw)
        assert "Syntax Error" not in cleaned
        assert "line 1" not in cleaned
        assert "column names" in cleaned.lower()

    def test_execution_timeout_cleaned(self):
        raw = "Execution Timeout: Code took longer than 30 seconds."
        cleaned = clean_response_answer(raw)
        assert "Execution Timeout" not in cleaned
        assert "longer than expected" in cleaned.lower()

    def test_valid_answer_untouched(self):
        answer = "Based on the report, net profits increased by $2.4M."
        assert clean_response_answer(answer) == answer


class TestFastAPIExceptionHandlers:
    """Integration tests for FastAPI application exception handlers."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import create_app
        app = create_app()
        return TestClient(app)

    def test_query_validation_error_is_friendly(self, client):
        # Sending empty body to /api/v1/query should trigger RequestValidationError on missing 'query'
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert "Please type a question or prompt before submitting." in data["detail"]
        assert "Field required" not in data["detail"]
        assert "loc" not in data["detail"]

    def test_invalid_uuid_param_is_friendly(self, client):
        # Sending invalid UUID to conversation endpoint
        response = client.get("/api/v1/conversations/not-a-valid-uuid")
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert "identifier is not recognized" in data["detail"] or "invalid" in data["detail"]
        assert "uuid_parsing" not in data["detail"]

