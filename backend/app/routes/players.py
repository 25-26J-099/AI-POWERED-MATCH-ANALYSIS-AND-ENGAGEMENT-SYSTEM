"""Player-related API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.models.models import Player, PlayerStats, PlayerEmbedding, Event
from app.analytics.heatmap import compute_heatmap_from_events

router = APIRouter()


@router.get("/match/{match_id}/players")
async def list_match_players(match_id: int, db: AsyncSession = Depends(get_db)):
    """List all players that appeared in a match."""
    result = await db.execute(
        select(PlayerStats).where(PlayerStats.match_id == match_id)
    )
    stats = result.scalars().all()
    return [
        {
            "player_id": s.player_id,
            "name": s.player.name if s.player else f"Player {s.player_id}",
            "team": s.player.team.name if s.player and s.player.team else None,
            "position": s.player.position if s.player else None,
            "rating": round(s.rating, 2),
        }
        for s in stats
    ]


@router.get("/match/{match_id}/player/{player_id}")
async def get_player_detail(match_id: int, player_id: int, db: AsyncSession = Depends(get_db)):
    """Get full player analysis: stats, heatmap, rating, and style cluster."""
    # Stats
    result = await db.execute(
        select(PlayerStats).where(
            PlayerStats.match_id == match_id,
            PlayerStats.player_id == player_id,
        )
    )
    stats = result.scalar_one_or_none()
    if not stats:
        raise HTTPException(status_code=404, detail="Player stats not found for this match")

    # Heatmap
    events_result = await db.execute(
        select(Event).where(
            Event.match_id == match_id,
            Event.player_id == player_id,
        )
    )
    events = events_result.scalars().all()
    heatmap = compute_heatmap_from_events(events)

    # Embedding / cluster
    emb_result = await db.execute(
        select(PlayerEmbedding).where(PlayerEmbedding.player_id == player_id)
    )
    embedding = emb_result.scalar_one_or_none()

    return {
        "player_id": player_id,
        "name": stats.player.name if stats.player else f"Player {player_id}",
        "team": stats.player.team.name if stats.player and stats.player.team else None,
        "position": stats.player.position if stats.player else None,
        "stats": {
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
        },
        "rating": round(stats.rating, 2),
        "heatmap": heatmap,
        "style_cluster": embedding.style_cluster if embedding else None,
        "umap": {"x": embedding.umap_x, "y": embedding.umap_y} if embedding else None,
    }


@router.get("/match/{match_id}/player/{player_id}/heatmap")
async def get_player_heatmap(match_id: int, player_id: int, db: AsyncSession = Depends(get_db)):
    """Get player heatmap data for this match."""
    result = await db.execute(
        select(Event).where(
            Event.match_id == match_id,
            Event.player_id == player_id,
        )
    )
    events = result.scalars().all()
    return {"player_id": player_id, "heatmap": compute_heatmap_from_events(events)}


@router.get("/match/{match_id}/player-comparison")
async def compare_players(
    match_id: int,
    player1: int = Query(..., description="First player ID"),
    player2: int = Query(..., description="Second player ID"),
    db: AsyncSession = Depends(get_db),
):
    """Compare two players side-by-side."""
    result1 = await db.execute(
        select(PlayerStats).where(PlayerStats.match_id == match_id, PlayerStats.player_id == player1)
    )
    result2 = await db.execute(
        select(PlayerStats).where(PlayerStats.match_id == match_id, PlayerStats.player_id == player2)
    )
    s1 = result1.scalar_one_or_none()
    s2 = result2.scalar_one_or_none()
    if not s1 or not s2:
        raise HTTPException(status_code=404, detail="One or both players not found")

    def to_radar(s: PlayerStats):
        return {
            "player_id": s.player_id,
            "name": s.player.name if s.player else f"Player {s.player_id}",
            "passes": s.passes,
            "pass_accuracy": round(s.pass_accuracy, 2),
            "progressive_passes": s.progressive_passes,
            "carries": s.carries,
            "shots": s.shots,
            "touches": s.touches,
            "pressures": s.pressures,
            "xg": round(s.xg, 4),
            "xt": round(s.xt, 4),
            "vaep": round(s.vaep, 4),
            "rating": round(s.rating, 2),
        }

    return {"player1": to_radar(s1), "player2": to_radar(s2)}
