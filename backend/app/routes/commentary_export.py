"""Commentary export endpoint for external LLM components."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.models.models import Event
from app.routes.matches import aggregate_team_stats, build_match_analytics

router = APIRouter()


@router.get("/match/{match_id}/analytics-export")
async def get_match_analytics_export(match_id: int, db: AsyncSession = Depends(get_db)):
    analytics = await build_match_analytics(match_id, db)
    result = await db.execute(
        select(Event).where(Event.match_id == match_id).order_by(Event.period, Event.minute, Event.second)
    )
    events = result.scalars().all()

    team_stats = {
        "home": aggregate_team_stats(
            [stats for stats in analytics["player_stats"] if stats.get("team") == analytics.get("home_team")]
        ),
        "away": aggregate_team_stats(
            [stats for stats in analytics["player_stats"] if stats.get("team") == analytics.get("away_team")]
        ),
    }

    return {
        "match_id": match_id,
        "events": [event.raw_data for event in events if event.raw_data],
        "xT": {
            "home": analytics["home_team_stats"]["total_xt"],
            "away": analytics["away_team_stats"]["total_xt"],
        },
        "xG": {
            "home": analytics["home_team_stats"]["total_xg"],
            "away": analytics["away_team_stats"]["total_xg"],
        },
        "VAEP": {
            "home": analytics["home_team_stats"]["total_vaep"],
            "away": analytics["away_team_stats"]["total_vaep"],
        },
        "player_stats": analytics["player_stats"],
        "team_stats": team_stats,
    }
