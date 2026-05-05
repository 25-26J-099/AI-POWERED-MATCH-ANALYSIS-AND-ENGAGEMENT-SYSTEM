"""Decision Quality API endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.analytics.decision_quality import compute_match_decision_quality
from app.database.database import get_db
from app.models.models import Event

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/match/{match_id}/decision-quality")
async def get_decision_quality(match_id: int, db: AsyncSession = Depends(get_db)):
    """Compute and return Decision Quality scores for all players in a match.

    Returns:
        players        — per-player DQ scores (0-100), tier, action count
        best_decisions — top 3 individual decisions
        worst_decisions — bottom 3 individual decisions
        total_events_analyzed — events that had enough freeze-frame data
    """
    result = await db.execute(
        select(Event)
        .where(Event.match_id == match_id)
        .order_by(Event.period, Event.minute, Event.second)
    )
    events = result.scalars().all()

    if not events:
        raise HTTPException(status_code=404, detail="No events found for this match")

    logger.info("Computing DQ for match_id=%s (%s events)", match_id, len(events))

    try:
        dq_data = compute_match_decision_quality(events)
    except Exception as exc:
        logger.exception("DQ computation failed for match_id=%s: %s", match_id, exc)
        raise HTTPException(status_code=500, detail=f"DQ computation error: {exc}") from exc

    return dq_data
