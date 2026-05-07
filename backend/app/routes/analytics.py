"""Analytics endpoints for matches and commentary consumers."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.routes.matches import build_match_analytics

router = APIRouter()


@router.get("/match/{match_id}/analytics")
async def get_match_analytics(match_id: int, db: AsyncSession = Depends(get_db)):
    return await build_match_analytics(match_id, db)
