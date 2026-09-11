"""Authenticated usage and cost reporting endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from backend.api.auth import get_user_key, is_authenticated
from backend.config import get_settings
from backend.database.connection import get_db
from backend.database.models import UsageEvent

router = APIRouter(prefix="/api/v1/usage", tags=["Usage"])


async def enforce_usage_limits(db: AsyncSession, owner_key: str) -> None:
    """Reject new work when configured rolling usage limits are exceeded."""
    settings = get_settings()
    if owner_key == "system:evaluator":
        return
    token_limit = settings.usage_max_tokens
    cost_limit = settings.usage_max_cost_usd
    if token_limit <= 0 and cost_limit <= 0:
        return

    since = datetime.now(timezone.utc) - timedelta(days=settings.usage_window_days)
    result = await db.execute(
        select(
            func.coalesce(func.sum(UsageEvent.total_tokens), 0),
            func.coalesce(func.sum(UsageEvent.estimated_cost_usd), 0.0),
        ).where(
            UsageEvent.owner_key == owner_key,
            UsageEvent.created_at >= since,
        )
    )
    total_tokens, total_cost = result.one()
    if token_limit > 0 and total_tokens >= token_limit:
        raise HTTPException(status_code=429, detail="Token usage limit reached for the current usage window")
    if cost_limit > 0 and float(total_cost) >= cost_limit:
        raise HTTPException(status_code=429, detail="Estimated cost limit reached for the current usage window")


@router.get("/summary")
async def usage_summary(
    request: Request,
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    """Return owner-scoped token and cost totals for the requested period."""
    if not await is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required for usage access")

    owner_key = await get_user_key(request)
    since = datetime.now(timezone.utc) - timedelta(days=days)
    filters = [UsageEvent.owner_key == owner_key, UsageEvent.created_at >= since]

    totals = await db.execute(
        select(
            func.count(UsageEvent.id),
            func.coalesce(func.sum(UsageEvent.input_tokens), 0),
            func.coalesce(func.sum(UsageEvent.output_tokens), 0),
            func.coalesce(func.sum(UsageEvent.total_tokens), 0),
            func.coalesce(func.sum(UsageEvent.estimated_cost_usd), 0.0),
        ).where(*filters)
    )
    count, input_tokens, output_tokens, total_tokens, cost = totals.one()

    grouped = await db.execute(
        select(
            UsageEvent.provider,
            UsageEvent.model,
            func.count(UsageEvent.id),
            func.sum(UsageEvent.total_tokens),
            func.sum(UsageEvent.estimated_cost_usd),
        ).where(*filters).group_by(UsageEvent.provider, UsageEvent.model)
    )

    return {
        "days": days,
        "query_count": count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(float(cost), 8),
        "limits": {
            "max_tokens": get_settings().usage_max_tokens,
            "max_cost_usd": get_settings().usage_max_cost_usd,
        },
        "by_model": [
            {
                "provider": provider,
                "model": model,
                "query_count": model_count,
                "total_tokens": model_tokens or 0,
                "estimated_cost_usd": round(float(model_cost or 0), 8),
            }
            for provider, model, model_count, model_tokens, model_cost in grouped.all()
        ],
    }