"""Match-related API endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.models.models import Match, PlayerStats, Event

router = APIRouter()


@router.get("/matches")
async def list_matches(db: AsyncSession = Depends(get_db)):
    """List all matches."""
    result = await db.execute(select(Match).order_by(Match.created_at.desc()))
    matches = result.scalars().all()
    return [
        {
            "id": m.id,
            "home_team": m.home_team.name if m.home_team else None,
            "away_team": m.away_team.name if m.away_team else None,
            "date": str(m.match_date) if m.match_date else None,
            "status": m.status,
            "created_at": m.created_at.isoformat(),
        }
        for m in matches
    ]


@router.get("/match/{match_id}")
async def get_match(match_id: int, db: AsyncSession = Depends(get_db)):
    """Get match details."""
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return {
        "id": match.id,
        "home_team": match.home_team.name if match.home_team else None,
        "away_team": match.away_team.name if match.away_team else None,
        "date": str(match.match_date) if match.match_date else None,
        "video_path": match.video_path,
        "commentary_video_path": match.commentary_video_path,
        "status": match.status,
        "status_detail": match.status_detail,
        "created_at": match.created_at.isoformat(),
    }


@router.get("/match/{match_id}/status")
async def get_match_status(match_id: int, db: AsyncSession = Depends(get_db)):
    """Get pipeline processing status for real-time updates."""
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return {
        "match_id": match.id,
        "status": match.status,
        "status_detail": match.status_detail,
    }


@router.get("/match/{match_id}/events")
async def get_match_events(match_id: int, db: AsyncSession = Depends(get_db)):
    """Get match event timeline."""
    result = await db.execute(
        select(Event)
        .where(Event.match_id == match_id)
        .order_by(Event.period, Event.minute, Event.second)
    )
    events = result.scalars().all()
    return [
        {
            "id": e.id,
            "event_uuid": e.event_uuid,
            "type": e.event_type,
            "player": e.player.name if e.player else None,
            "team": e.team.name if e.team else None,
            "minute": e.minute,
            "second": e.second,
            "period": e.period,
            "x": e.x,
            "y": e.y,
            "end_x": e.end_x,
            "end_y": e.end_y,
        }
        for e in events
    ]


@router.get("/match/{match_id}/analytics")
async def get_match_analytics(match_id: int, db: AsyncSession = Depends(get_db)):
    """Full analytics payload — used by commentary LLM and the AI analysis page."""
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    # Player stats
    stats_result = await db.execute(
        select(PlayerStats).where(PlayerStats.match_id == match_id)
    )
    all_stats = stats_result.scalars().all()

    player_stats_data = []
    for s in all_stats:
        player_stats_data.append({
            "player_id": s.player_id,
            "player_name": s.player.name if s.player else f"Player {s.player_id}",
            "team": s.player.team.name if s.player and s.player.team else None,
            "passes": s.passes,
            "pass_accuracy": round(s.pass_accuracy, 2),
            "progressive_passes": s.progressive_passes,
            "carries": s.carries,
            "shots": s.shots,
            "touches": s.touches,
            "pressures": s.pressures,
            "recoveries": s.recoveries,
            "xg": round(s.xg, 4),
            "xt": round(s.xt, 4),
            "vaep": round(s.vaep, 4),
            "rating": round(s.rating, 2),
        })

    # Team aggregates
    home_stats = [s for s in player_stats_data if s.get("team") == (match.home_team.name if match.home_team else None)]
    away_stats = [s for s in player_stats_data if s.get("team") == (match.away_team.name if match.away_team else None)]

    def aggregate_team(stats_list):
        return {
            "total_xg": round(sum(s["xg"] for s in stats_list), 4),
            "total_xt": round(sum(s["xt"] for s in stats_list), 4),
            "total_vaep": round(sum(s["vaep"] for s in stats_list), 4),
            "total_passes": sum(s["passes"] for s in stats_list),
            "avg_pass_accuracy": round(sum(s["pass_accuracy"] for s in stats_list) / max(len(stats_list), 1), 2),
            "total_shots": sum(s["shots"] for s in stats_list),
        }

    return {
        "match_id": match_id,
        "home_team": match.home_team.name if match.home_team else None,
        "away_team": match.away_team.name if match.away_team else None,
        "player_stats": player_stats_data,
        "home_team_stats": aggregate_team(home_stats),
        "away_team_stats": aggregate_team(away_stats),
    }
