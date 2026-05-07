"""Player-related API endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.analytics.heatmap import compute_heatmap_from_events
from app.analytics.style_clusters import get_style_cluster_label
from app.database.database import get_db
from app.models.models import Event, PlayerEmbedding, PlayerStats

router = APIRouter()


def _serialize_player_stats(stats: PlayerStats) -> dict:
    return {
        "player_id": stats.player_id,
        "name": stats.player.name if stats.player else f"Player {stats.player_id}",
        "team": stats.player.team.name if stats.player and stats.player.team else None,
        "position": stats.player.position if stats.player else None,
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
        "match_id": stats.match_id,
        "style_cluster": None,
        "style_cluster_label": None,
    }


async def _get_stats_for_match(match_id: int, player_id: int, db: AsyncSession) -> PlayerStats:
    result = await db.execute(
        select(PlayerStats).where(PlayerStats.match_id == match_id, PlayerStats.player_id == player_id)
    )
    stats = result.scalar_one_or_none()
    if stats is None:
        raise HTTPException(status_code=404, detail="Player stats not found for this match")
    return stats


async def _get_latest_stats(player_id: int, db: AsyncSession) -> PlayerStats:
    result = await db.execute(
        select(PlayerStats).where(PlayerStats.player_id == player_id).order_by(desc(PlayerStats.match_id))
    )
    stats = result.scalars().first()
    if stats is None:
        raise HTTPException(status_code=404, detail="Player stats not found")
    return stats


async def _get_player_embedding(player_id: int, match_id: int | None, db: AsyncSession) -> PlayerEmbedding | None:
    query = select(PlayerEmbedding).where(PlayerEmbedding.player_id == player_id)
    if match_id is not None:
        query = query.where(PlayerEmbedding.match_id == match_id)
    query = query.order_by(desc(PlayerEmbedding.match_id))
    result = await db.execute(query)
    return result.scalars().first()


@router.get("/match/{match_id}/players")
async def list_match_players(match_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(PlayerStats).where(PlayerStats.match_id == match_id))
    stats = result.scalars().all()
    return [
        {
            "player_id": stat.player_id,
            "name": stat.player.name if stat.player else f"Player {stat.player_id}",
            "team": stat.player.team.name if stat.player and stat.player.team else None,
            "position": stat.player.position if stat.player else None,
            "rating": round(stat.rating, 2),
        }
        for stat in stats
    ]


@router.get("/match/{match_id}/player/{player_id}")
async def get_player_detail(match_id: int, player_id: int, db: AsyncSession = Depends(get_db)):
    stats = await _get_stats_for_match(match_id, player_id, db)
    events_result = await db.execute(select(Event).where(Event.match_id == match_id, Event.player_id == player_id))
    events = events_result.scalars().all()
    heatmap = compute_heatmap_from_events(events)
    embedding = await _get_player_embedding(player_id, match_id, db)

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
        "style_cluster_label": get_style_cluster_label(embedding.style_cluster) if embedding else None,
        "umap": {"x": embedding.umap_x, "y": embedding.umap_y} if embedding else None,
    }


@router.get("/match/{match_id}/player/{player_id}/heatmap")
async def get_match_player_heatmap(match_id: int, player_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Event).where(Event.match_id == match_id, Event.player_id == player_id))
    events = result.scalars().all()
    return {"player_id": player_id, "match_id": match_id, "heatmap": compute_heatmap_from_events(events)}


@router.get("/match/{match_id}/player-comparison")
async def compare_match_players(
    match_id: int,
    player1: int = Query(..., description="First player ID"),
    player2: int = Query(..., description="Second player ID"),
    db: AsyncSession = Depends(get_db),
):
    stat1 = await _get_stats_for_match(match_id, player1, db)
    stat2 = await _get_stats_for_match(match_id, player2, db)
    return {"player1": _serialize_player_stats(stat1), "player2": _serialize_player_stats(stat2)}


@router.get("/player/{player_id}/stats")
async def get_player_stats(player_id: int, db: AsyncSession = Depends(get_db)):
    stats = await _get_latest_stats(player_id, db)
    return _serialize_player_stats(stats)


@router.get("/player/{player_id}/heatmap")
async def get_player_heatmap(player_id: int, match_id: int | None = None, db: AsyncSession = Depends(get_db)):
    stats = await (_get_stats_for_match(match_id, player_id, db) if match_id is not None else _get_latest_stats(player_id, db))
    result = await db.execute(select(Event).where(Event.match_id == stats.match_id, Event.player_id == player_id))
    events = result.scalars().all()
    return {"player_id": player_id, "match_id": stats.match_id, "heatmap": compute_heatmap_from_events(events)}


@router.get("/player/{player_id}/style")
async def get_player_style(player_id: int, match_id: int | None = None, db: AsyncSession = Depends(get_db)):
    stats = await (_get_stats_for_match(match_id, player_id, db) if match_id is not None else _get_latest_stats(player_id, db))
    embedding = await _get_player_embedding(player_id, stats.match_id, db)
    if embedding is None:
        raise HTTPException(status_code=404, detail="Player style embedding not found")
    return {
        "player_id": player_id,
        "match_id": stats.match_id,
        "cluster": embedding.style_cluster,
        "cluster_label": get_style_cluster_label(embedding.style_cluster),
        "embedding_vector": embedding.embedding_vector,
        "umap": {"x": embedding.umap_x, "y": embedding.umap_y},
        "tsne": {"x": embedding.tsne_x, "y": embedding.tsne_y},
    }


@router.get("/player/{player_id}/comparison/{player2_id}")
async def compare_players(player_id: int, player2_id: int, match_id: int | None = None, db: AsyncSession = Depends(get_db)):
    if match_id is not None:
        stat1 = await _get_stats_for_match(match_id, player_id, db)
        stat2 = await _get_stats_for_match(match_id, player2_id, db)
    else:
        stat1 = await _get_latest_stats(player_id, db)
        stat2 = await _get_latest_stats(player2_id, db)
    return {"player1": _serialize_player_stats(stat1), "player2": _serialize_player_stats(stat2)}
