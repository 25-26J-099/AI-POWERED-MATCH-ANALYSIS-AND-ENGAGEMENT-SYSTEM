"""Merged upload entrypoint that launches tracking and analytics."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.database.database import get_db
from app.models.models import Match
from app.services.dependencies import get_job_service
from app.services.merged_pipeline_service import process_match_video

router = APIRouter()


@router.post("/upload-video")
async def upload_video(
    video: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a video and create a match record (pipeline starts after lineup setup)."""
    filename = video.filename or "uploaded_video.mp4"
    ext = Path(filename).suffix or ".mp4"
    stored_name = f"{uuid.uuid4().hex}{ext}"
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / stored_name

    try:
        with video_path.open("wb") as outfile:
            while chunk := await video.read(1024 * 1024):
                outfile.write(chunk)
    finally:
        await video.close()

    job_service = get_job_service()
    job = job_service.create_job()

    match = Match(
        video_path=str(video_path),
        tracking_job_id=job.job_id,
        status="lineup_pending",
        status_detail="Video uploaded. Waiting for lineup setup...",
    )
    db.add(match)
    await db.commit()
    await db.refresh(match)

    return {
        "match_id": match.id,
        "job_id": job.job_id,
        "status": match.status,
        "message": "Video uploaded. Proceed to lineup setup.",
    }


@router.post("/match/{match_id}/proceed")
async def proceed_pipeline(match_id: int, db: AsyncSession = Depends(get_db)):
    """Start the merged tracking + analytics pipeline after lineup setup."""
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    if match.status not in ("lineup_pending", "uploading"):
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline already started or completed (status: {match.status})",
        )

    match.status = "uploading"
    match.status_detail = "Lineup submitted. Starting tracking pipeline..."
    await db.commit()

    asyncio.create_task(
        process_match_video(match.id, match.video_path, match.tracking_job_id)
    )

    return {
        "match_id": match.id,
        "status": "uploading",
        "message": "Pipeline started.",
    }

