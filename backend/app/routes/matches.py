"""Match-related API endpoints and shared analytics helpers."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.models.models import Event, Match, PlayerStats
from app.services.dependencies import get_job_service

router = APIRouter()


async def get_match_or_404(match_id: int, db: AsyncSession) -> Match:
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")
    return match


def aggregate_team_stats(stats_list: list[dict]) -> dict:
    return {
        "total_xg": round(sum(s["xg"] for s in stats_list), 4),
        "total_xt": round(sum(s["xt"] for s in stats_list), 4),
        "total_vaep": round(sum(s["vaep"] for s in stats_list), 4),
        "total_passes": sum(s["passes"] for s in stats_list),
        "avg_pass_accuracy": round(sum(s["pass_accuracy"] for s in stats_list) / max(len(stats_list), 1), 2),
        "total_shots": sum(s["shots"] for s in stats_list),
    }


async def build_match_analytics(match_id: int, db: AsyncSession) -> dict:
    match = await get_match_or_404(match_id, db)
    stats_result = await db.execute(select(PlayerStats).where(PlayerStats.match_id == match_id))
    all_stats = stats_result.scalars().all()

    player_stats_data = []
    for stats in all_stats:
        player_stats_data.append(
            {
                "player_id": stats.player_id,
                "player_name": stats.player.name if stats.player else f"Player {stats.player_id}",
                "team": stats.player.team.name if stats.player and stats.player.team else None,
                "passes": stats.passes,
                "pass_accuracy": round(stats.pass_accuracy, 2),
                "progressive_passes": stats.progressive_passes,
                "carries": stats.carries,
                "shots": stats.shots,
                "touches": stats.touches,
                "pressures": stats.pressures,
                "recoveries": stats.recoveries,
                "xg": round(stats.xg, 4),
                "xt": round(stats.xt, 4),
                "vaep": round(stats.vaep, 4),
                "rating": round(stats.rating, 2),
            }
        )

    home_team_name = match.home_team.name if match.home_team else None
    away_team_name = match.away_team.name if match.away_team else None
    home_stats = [stats for stats in player_stats_data if stats.get("team") == home_team_name]
    away_stats = [stats for stats in player_stats_data if stats.get("team") == away_team_name]

    return {
        "match_id": match_id,
        "home_team": home_team_name,
        "away_team": away_team_name,
        "player_stats": player_stats_data,
        "home_team_stats": aggregate_team_stats(home_stats),
        "away_team_stats": aggregate_team_stats(away_stats),
    }


@router.get("/matches")
async def list_matches(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Match).order_by(Match.created_at.desc()))
    matches = result.scalars().all()
    return [
        {
            "id": match.id,
            "home_team": match.home_team.name if match.home_team else None,
            "away_team": match.away_team.name if match.away_team else None,
            "date": str(match.match_date) if match.match_date else None,
            "status": match.status,
            "created_at": match.created_at.isoformat(),
        }
        for match in matches
    ]


@router.get("/match/{match_id}")
async def get_match(match_id: int, db: AsyncSession = Depends(get_db)):
    match = await get_match_or_404(match_id, db)
    return {
        "id": match.id,
        "home_team": match.home_team.name if match.home_team else None,
        "away_team": match.away_team.name if match.away_team else None,
        "date": str(match.match_date) if match.match_date else None,
        "video_path": match.video_path,
        "commentary_video_path": match.commentary_video_path,
        "tracking_job_id": match.tracking_job_id,
        "tracking_artifacts": match.tracking_artifacts or {},
        "status": match.status,
        "status_detail": match.status_detail,
        "created_at": match.created_at.isoformat(),
    }


@router.get("/match/{match_id}/status")
async def get_match_status(match_id: int, db: AsyncSession = Depends(get_db)):
    match = await get_match_or_404(match_id, db)
    tracking_job_status = None
    tracking_job_error = None
    if match.tracking_job_id:
        try:
            job = get_job_service().get_job(match.tracking_job_id)
            tracking_job_status = job.status
            tracking_job_error = job.error
        except KeyError:
            tracking_job_status = "missing"
    return {
        "match_id": match.id,
        "job_id": match.tracking_job_id,
        "status": match.status,
        "status_detail": match.status_detail,
        "tracking_job_status": tracking_job_status,
        "tracking_job_error": tracking_job_error,
    }


@router.get("/match/{match_id}/events")
async def get_match_events(match_id: int, db: AsyncSession = Depends(get_db)):
    await get_match_or_404(match_id, db)
    result = await db.execute(
        select(Event).where(Event.match_id == match_id).order_by(Event.period, Event.minute, Event.second)
    )
    events = result.scalars().all()
    return [
        {
            "id": event.id,
            "event_uuid": event.event_uuid,
            "type": event.event_type,
            "player": event.player.name if event.player else None,
            "team": event.team.name if event.team else None,
            "minute": event.minute,
            "second": event.second,
            "period": event.period,
            "timestamp": event.timestamp,
            "x": event.x,
            "y": event.y,
            "end_x": event.end_x,
            "end_y": event.end_y,
            "freeze_frame": event.freeze_frame,
            "raw_data": event.raw_data,
        }
        for event in events
    ]
