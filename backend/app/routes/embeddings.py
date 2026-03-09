"""Player style embedding endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.models.models import PlayerEmbedding, Player

router = APIRouter()


@router.get("/match/{match_id}/style-map")
async def get_style_map(match_id: int, db: AsyncSession = Depends(get_db)):
    """Get UMAP/t-SNE reduced embeddings for player style scatter plot."""
    result = await db.execute(
        select(PlayerEmbedding).where(PlayerEmbedding.match_id == match_id)
    )
    embeddings = result.scalars().all()
    if not embeddings:
        raise HTTPException(status_code=404, detail="No embeddings computed for this match yet")

    return {
        "match_id": match_id,
        "players": [
            {
                "player_id": e.player_id,
                "name": e.player.name if e.player else f"Player {e.player_id}",
                "team": e.player.team.name if e.player and e.player.team else None,
                "umap_x": e.umap_x,
                "umap_y": e.umap_y,
                "tsne_x": e.tsne_x,
                "tsne_y": e.tsne_y,
                "cluster": e.style_cluster,
            }
            for e in embeddings
        ],
    }
