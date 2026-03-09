"""Video upload endpoint — triggers the full analysis pipeline."""

import os
import uuid
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.database.database import get_db
from app.models.models import Match
from app.services.pipeline import run_full_pipeline

router = APIRouter()


@router.post("/upload-video")
async def upload_video(
    video: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
):
    """Upload a match video and automatically start the analysis pipeline.

    The pipeline runs as a background task:
    Upload → Tracking → Event Detection → Analytics → Commentary → Done
    """
    # Generate unique filename
    ext = os.path.splitext(video.filename or "video.mp4")[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    video_path = os.path.join(settings.UPLOAD_DIR, filename)

    # Save video file
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    contents = await video.read()
    with open(video_path, "wb") as f:
        f.write(contents)

    # Create match record
    match = Match(
        video_path=video_path,
        status="uploaded",
        status_detail="Video uploaded successfully. Waiting for lineups.",
    )
    db.add(match)
    await db.flush()
    await db.refresh(match)
    match_id = match.id

    return {
        "match_id": match_id,
        "status": "uploaded",
        "message": "Video uploaded. Submit lineups and click Proceed to start pipeline.",
        "video_path": video_path,
    }


@router.post("/match/{match_id}/proceed")
async def proceed_pipeline(
    match_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Start the full analysis pipeline for a match (after lineups are submitted)."""
    from sqlalchemy import select

    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    if not match:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Match not found")

    match.status = "tracking"
    match.status_detail = "Pipeline started. Sending video to tracking component..."
    await db.commit()

    # Launch full pipeline in background
    background_tasks.add_task(run_full_pipeline, match_id)

    return {"match_id": match_id, "status": "tracking", "message": "Pipeline started."}
