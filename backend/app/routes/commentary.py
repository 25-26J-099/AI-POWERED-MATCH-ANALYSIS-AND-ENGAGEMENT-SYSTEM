from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import FileResponse

from app.database.database import get_db
from app.models.models import Match, CommentaryOutput
from app.services.commentary_service import generate_commentary
from app.services.security import require_enterprise_api_key

router = APIRouter(prefix="/match/{match_id}/commentary", tags=["commentary"])

@router.get("/")
async def get_commentary_outputs(match_id: int, db: AsyncSession = Depends(get_db)):
    """Retrieve all commentary texts generated for a match."""
    result = await db.execute(select(CommentaryOutput).where(CommentaryOutput.match_id == match_id))
    outputs = result.scalars().all()
    if not outputs:
        return []
    return [{"id": o.id, "type": o.commentary_type, "text": o.commentary_text, "video_path": o.video_path} for o in outputs]

@router.get("/video")
async def get_commentary_video(match_id: int, db: AsyncSession = Depends(get_db)):
    """Stream the generated commentary video for the match."""
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
        
    if not match.commentary_video_path:
        raise HTTPException(status_code=404, detail="Commentary video not generated yet")
        
    return FileResponse(
        match.commentary_video_path,
        media_type="video/mp4",
        filename=f"match_{match_id}_commentary.mp4"
    )

@router.post("/trigger", dependencies=[Depends(require_enterprise_api_key)])
async def trigger_commentary(match_id: int, db: AsyncSession = Depends(get_db)):
    """Manually trigger the commentary pipeline for a match."""
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
        
    # Start generation in background (or run inline depending on async context)
    # generate_commentary handles its own session inside async def so we just start task
    import asyncio
    asyncio.create_task(generate_commentary(match_id))
    return {"status": "accepted", "message": "Commentary generation triggered in background."}
