"""Lineup management endpoints — formation selection and player input."""

from typing import List
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.models.models import Lineup, LineupPlayer, Team, Match

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────


class LineupPlayerInput(BaseModel):
    player_name: str
    jersey_number: int
    position_slot: int = Field(..., ge=0, le=10, description="Index in formation (0=GK, then outfield)")


class LineupInput(BaseModel):
    team_name: str
    formation: str = Field(..., pattern=r"^\d-\d(-\d){1,3}$", description="e.g. '4-3-3', '4-4-2', '3-5-2'")
    players: List[LineupPlayerInput] = Field(..., min_length=11, max_length=11)


class SubmitLineupsRequest(BaseModel):
    home_team: LineupInput
    away_team: LineupInput


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.post("/match/{match_id}/lineups")
async def submit_lineups(
    match_id: int,
    body: SubmitLineupsRequest,
    db: AsyncSession = Depends(get_db),
):
    """Submit lineups for both teams (formation + 11 players each)."""
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    async def upsert_team(name: str) -> Team:
        res = await db.execute(select(Team).where(Team.name == name))
        team = res.scalar_one_or_none()
        if not team:
            team = Team(name=name)
            db.add(team)
            await db.flush()
        return team

    async def create_lineup(team: Team, data: LineupInput) -> Lineup:
        lineup = Lineup(match_id=match_id, team_id=team.id, formation=data.formation)
        db.add(lineup)
        await db.flush()
        for p in data.players:
            db.add(LineupPlayer(
                lineup_id=lineup.id,
                player_name=p.player_name,
                jersey_number=p.jersey_number,
                position_slot=p.position_slot,
            ))
        return lineup

    home_team = await upsert_team(body.home_team.team_name)
    away_team = await upsert_team(body.away_team.team_name)

    match.home_team_id = home_team.id
    match.away_team_id = away_team.id

    home_lineup = await create_lineup(home_team, body.home_team)
    away_lineup = await create_lineup(away_team, body.away_team)

    await db.commit()

    return {
        "match_id": match_id,
        "home_lineup_id": home_lineup.id,
        "away_lineup_id": away_lineup.id,
        "message": "Lineups submitted successfully.",
    }


@router.get("/match/{match_id}/lineups")
async def get_lineups(match_id: int, db: AsyncSession = Depends(get_db)):
    """Retrieve lineups for a match."""
    result = await db.execute(
        select(Lineup).where(Lineup.match_id == match_id)
    )
    lineups = result.scalars().all()
    if not lineups:
        raise HTTPException(status_code=404, detail="No lineups found for this match")

    return [
        {
            "id": l.id,
            "team": l.team.name if l.team else None,
            "formation": l.formation,
            "players": [
                {
                    "player_name": p.player_name,
                    "jersey_number": p.jersey_number,
                    "position_slot": p.position_slot,
                }
                for p in sorted(l.players, key=lambda x: x.position_slot)
            ],
        }
        for l in lineups
    ]
