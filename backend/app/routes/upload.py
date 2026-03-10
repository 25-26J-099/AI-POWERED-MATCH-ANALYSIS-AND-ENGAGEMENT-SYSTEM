"""Merged upload entrypoint that launches tracking and analytics."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
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
    """Upload a video and immediately start the merged processing pipeline."""
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
        status="uploading",
        status_detail="Upload received. Scheduling tracking pipeline...",
    )
    db.add(match)
    await db.commit()
    await db.refresh(match)

    asyncio.create_task(process_match_video(match.id, str(video_path), job.job_id))

    return {
        "match_id": match.id,
        "job_id": job.job_id,
        "status": match.status,
        "message": "Video uploaded and processing started.",
    }


@router.post("/match/{match_id}/proceed")
async def proceed_pipeline(match_id: int):
    """Legacy compatibility endpoint kept for the existing frontend flow."""
    raise HTTPException(
        status_code=409,
        detail="Pipeline starts automatically after POST /upload-video in the merged system.",
    )
