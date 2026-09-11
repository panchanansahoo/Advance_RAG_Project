"""Tests for usage aggregation and quota enforcement."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from backend.api.usage import enforce_usage_limits, usage_summary


def _settings(**overrides):
    values = {
        "usage_window_days": 30,
        "usage_max_tokens": 0,
        "usage_max_cost_usd": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_unlimited_usage_does_not_query_database():
    db = MagicMock()
    with patch("backend.api.usage.get_settings", return_value=_settings()):
        await enforce_usage_limits(db, "session:user@example.com")

    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_token_limit_rejects_at_limit():
    db = MagicMock()
    result = MagicMock()
    result.one.return_value = (1000, 0.25)
    db.execute = AsyncMock(return_value=result)

    with patch(
        "backend.api.usage.get_settings",
        return_value=_settings(usage_max_tokens=1000),
    ):
        with pytest.raises(HTTPException, match="Token usage limit"):
            await enforce_usage_limits(db, "session:user@example.com")


@pytest.mark.asyncio
async def test_cost_limit_rejects_at_limit():
    db = MagicMock()
    result = MagicMock()
    result.one.return_value = (100, 2.0)
    db.execute = AsyncMock(return_value=result)

    with patch(
        "backend.api.usage.get_settings",
        return_value=_settings(usage_max_cost_usd=2.0),
    ):
        with pytest.raises(HTTPException, match="Estimated cost limit"):
            await enforce_usage_limits(db, "session:user@example.com")


@pytest.mark.asyncio
async def test_evaluator_bypasses_usage_limits():
    db = MagicMock()
    with patch(
        "backend.api.usage.get_settings",
        return_value=_settings(usage_max_tokens=1, usage_max_cost_usd=0.01),
    ):
        await enforce_usage_limits(db, "system:evaluator")

    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_usage_summary_returns_totals_and_model_breakdown():
    db = MagicMock()
    totals = MagicMock()
    totals.one.return_value = (3, 120, 80, 200, 0.0042)
    grouped = MagicMock()
    grouped.all.return_value = [("openai", "gpt-4o-mini", 3, 200, 0.0042)]
    db.execute = AsyncMock(side_effect=[totals, grouped])
    request = MagicMock()

    with patch("backend.api.usage.get_settings", return_value=_settings()), \
         patch("backend.api.usage.is_authenticated", new=AsyncMock(return_value=True)), \
         patch("backend.api.usage.get_user_key", new=AsyncMock(return_value="session:user@example.com")):
        response = await usage_summary(request, days=30, db=db)

    assert response["query_count"] == 3
    assert response["total_tokens"] == 200
    assert response["estimated_cost_usd"] == 0.0042
    assert response["by_model"] == [{
        "provider": "openai",
        "model": "gpt-4o-mini",
        "query_count": 3,
        "total_tokens": 200,
        "estimated_cost_usd": 0.0042,
    }]