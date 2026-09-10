"""
Smart, user-friendly error sanitization and formatting.

Translates technical exceptions, tracebacks, database errors, validation errors,
and programming jargon into clear, polite, and actionable human-friendly messages.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Union


def format_validation_error(errors: Any) -> str:
    """
    Format FastAPI / Pydantic RequestValidationError into a polite, clean sentence.
    """
    if not isinstance(errors, list) or not errors:
        return "Please verify your input and try again."

    first_err = errors[0]
    loc = first_err.get("loc", [])
    msg = first_err.get("msg", "")
    err_type = first_err.get("type", "")

    # Extract field name
    field_name = ""
    if loc:
        if len(loc) > 1 and isinstance(loc[-1], str):
            field_name = loc[-1]
        elif isinstance(loc[0], str) and loc[0] not in ("body", "header", "path"):
            field_name = loc[0]

    # Specific friendly translations for common fields
    if field_name == "query":
        if "missing" in err_type or "required" in msg.lower():
            return "Please type a question or prompt before submitting."
        return "Please check the question you entered and try again."

    if field_name in ("file", "files"):
        return "Please select a valid document to upload."

    if field_name in ("conversation_id", "document_id", "id"):
        return "The requested document or conversation link is invalid. Please check the URL and try again."

    if "missing" in err_type or "required" in msg.lower():
        readable_field = field_name.replace("_", " ").strip() if field_name else "required information"
        return f"Please provide the {readable_field} to continue."

    if "uuid" in err_type or "uuid" in msg.lower():
        return "The requested identifier is not recognized. Please verify the link and try again."

    return "Some of the provided information is incomplete or invalid. Please check your entries and try again."


def sanitize_error_message(
    err: Any,
    default_fallback: str = "We encountered a temporary issue while processing your request. Please try again.",
) -> str:
    """
    Smartly converts any technical error string or exception into a user-friendly message.
    """
    if err is None:
        return default_fallback

    # Extract text representation
    if isinstance(err, list):
        return format_validation_error(err)
    elif isinstance(err, dict):
        if "detail" in err:
            return sanitize_error_message(err["detail"], default_fallback)
        if "message" in err:
            return sanitize_error_message(err["message"], default_fallback)
        raw_str = str(err)
    elif isinstance(err, Exception):
        raw_str = str(err)
    else:
        raw_str = str(err)

    if not raw_str or not raw_str.strip():
        return default_fallback

    cleaned = raw_str.strip()
    lower = cleaned.lower()

    # 1. Network / Connectivity / Server Offline (check before file/OS errors)
    if any(
        kw in lower
        for kw in (
            "failed to fetch",
            "networkerror",
            "econnrefused",
            "connection refused",
            "connection reset",
            "temporarily unavailable",
            "offline",
            "socket",
            "disconnected",
        )
    ):
        return (
            "The assistant service is momentarily unreachable. Please check your internet connection "
            "or try again in a moment."
        )

    # 2. File upload & format errors
    if "not supported" in lower or "allowed extensions" in lower or "allowed:" in lower:
        return (
            "This file type is not supported. Please upload a PDF, Word document (.docx), "
            "Excel spreadsheet (.xlsx, .csv), Markdown (.md), or plain text (.txt) file."
        )

    if "exceeds the maximum allowed size" in lower or "file size (" in lower:
        # Extract sizes if present
        size_match = re.search(r"\((\d+(?:\.\d+)?)\s*mb\)", cleaned, re.IGNORECASE)
        max_match = re.search(r"allowed size \((\d+(?:\.\d+)?)\s*mb\)", cleaned, re.IGNORECASE)
        if size_match and max_match:
            return (
                f"This file ({size_match.group(1)} MB) exceeds the maximum allowed size of "
                f"{max_match.group(1)} MB. Please upload a smaller file or split it into sections."
            )
        return "This file exceeds the maximum allowed upload size. Please upload a smaller file or split it into sections."

    if "file is empty" in lower or "empty file" in lower:
        return "The selected file appears to be empty. Please choose a file containing text or data."

    if "filename is required" in lower:
        return "Please select a file with a valid name to upload."

    if "failed to save file" in lower or "permission denied" in lower or "locked" in lower:
        return (
            "We were unable to save the uploaded file. Please verify that the file is not currently "
            "locked or open in another program and try again."
        )

    # 3. Timeouts
    if any(kw in lower for kw in ("timeout", "timed out", "asyncio.timeouterror")):
        return "The request took longer than expected to complete. Please try asking again or breaking your question into smaller parts."

    # 4. LLM / AI Provider rate limits, quota, and overloads
    if any(kw in lower for kw in ("rate limit", "quota", "429", "too many requests", "resourceexhausted")):
        return "The AI assistant is experiencing high demand right now. Please wait a moment before sending another query."

    if any(kw in lower for kw in ("api key", "unauthorized", "authenticationerror", "401", "403")):
        return "The AI assistant is temporarily unavailable due to a service configuration update. Please try again shortly."

    if any(kw in lower for kw in ("model is overloaded", "503", "service unavailable", "bad gateway", "502")):
        return "The AI model is currently busy. Please try sending your request again in a moment."

    if "context_length_exceeded" in lower or "maximum context length" in lower or "token limit" in lower:
        return "This question or document is too lengthy to analyze all at once. Please try asking about a specific section or summarizing."

    # 5. Resource not found (404)
    if "document not found" in lower:
        return "The requested document could not be found. It may have been removed or deleted."

    if "conversation not found" in lower:
        return "The requested conversation could not be found. It may have been cleared or deleted."

    if "not found" in lower or "404" in lower:
        return "The requested item could not be found."

    # 6. Spreadsheet / Tabular analysis errors
    if "import of" in lower and "not allowed" in lower:
        return "For security reasons, this specific calculation or operation cannot be performed on spreadsheet data."

    if "call to" in lower and "not allowed" in lower:
        return "For security reasons, this specific calculation or function cannot be performed on spreadsheet data."

    if any(
        kw in lower
        for kw in (
            "pandas",
            "dataframe",
            "keyerror",
            "syntax error in generated code",
            "execution error",
            "failed to read schemas",
            "failed to generate execution code",
        )
    ):
        return (
            "We were unable to complete the calculation on this spreadsheet. "
            "Please check that your question references columns present in the file, or try rephrasing."
        )

    # 7. Document processing / extraction failures
    if any(kw in lower for kw in ("pypdf", "docx", "pdfreaderror", "corrupt", "unreadable")):
        return "We were unable to extract text from this document. The file may be damaged or password-protected."

    # 8. Technical code signals: tracebacks, SQL, exceptions, AST, memory addresses
    has_tech_signals = (
        "traceback" in lower
        or "most recent call last" in lower
        or "exception" in lower
        or "error:" in lower
        or "syntaxerror" in lower
        or "valueerror" in lower
        or "typeerror" in lower
        or "attributeerror" in lower
        or "nameerror" in lower
        or "filenotfounderror" in lower
        or "internal server error" in lower
        or "sqlite3" in lower
        or "sqlalchemy" in lower
        or "psycopg2" in lower
        or "asyncpg" in lower
        or "status code" in lower
        or "fastapi" in lower
        or "uvicorn" in lower
        or "line " in lower
        or "<class" in lower
        or "0x" in lower
        or bool(re.search(r"[{}[\]<>()_\\/]{3,}", cleaned))
    )

    if has_tech_signals:
        return default_fallback

    return cleaned


def clean_response_answer(
    answer: str,
    default_fallback: str = "I couldn't generate a complete answer. Please try rephrasing your question or checking your uploaded documents.",
) -> str:
    """
    Sanitize an agent or LLM answer before returning it to the user.
    If the answer starts with an execution error, traceback, or technical failure,
    converts it into a polite, user-friendly explanation.
    """
    if not answer or not answer.strip():
        return default_fallback

    lower = answer.lower().strip()

    # Raw execution error from Pandas agent
    if lower.startswith("execution error:") or lower.startswith("syntax error in generated code:"):
        return (
            "I encountered an issue calculating the answer from this spreadsheet data. "
            "Please verify that the column names mentioned in your question are in the file, "
            "or try asking the question in a simpler way."
        )

    # Raw summarization failed
    if lower.startswith("raw result (summarization failed):"):
        return (
            "I analyzed the data, but had difficulty formatting a final summary. "
            "Please try asking your question again."
        )

    # Execution timeout
    if lower.startswith("execution timeout:"):
        return (
            "Analyzing this data took longer than expected. "
            "Please try asking a more focused question or checking a specific section of the data."
        )

    # Final answer error
    if "an error occurred while generating the final answer" in lower:
        return (
            "I ran into an issue putting together the final answer for your question. "
            "Please try asking again or rephrasing."
        )

    # General python traceback in answer
    if "traceback (most recent call last)" in lower:
        return (
            "I encountered an unexpected difficulty while analyzing the information. "
            "Please try rephrasing your question."
        )

    return answer
