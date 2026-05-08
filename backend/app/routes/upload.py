"""Merged upload entrypoint that launches tracking and analytics."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.commentary.adaptive import AUTO_COMMENTARY_LEVELS, COMMENTARY_STYLES, COMMENTARY_VERBOSITY
from app.config.settings import settings
from app.database.database import get_db
from app.models.models import Match, Team
from app.services.dependencies import get_job_service
from app.services.football_video_validator import validate_football_video
from app.services.merged_pipeline_service import process_match_video
from app.services.team_color_service import detect_team_colors_preview

router = APIRouter()

ALLOWED_COMMENTARY_LEVELS = ("Beginner", "Intermediate", "Expert", *AUTO_COMMENTARY_LEVELS)
_TEAM_COLOR_DETECTION_SEMAPHORE = asyncio.Semaphore(max(1, settings.TEAM_COLOR_DETECTION_MAX_CONCURRENCY))
_TEAM_COLOR_MATCH_LOCKS: dict[int, asyncio.Lock] = {}
_TEAM_COLOR_MATCH_LOCKS_GUARD = asyncio.Lock()


async def _get_team_color_match_lock(match_id: int) -> asyncio.Lock:
    async with _TEAM_COLOR_MATCH_LOCKS_GUARD:
        lock = _TEAM_COLOR_MATCH_LOCKS.get(match_id)
        if lock is None:
            lock = asyncio.Lock()
            _TEAM_COLOR_MATCH_LOCKS[match_id] = lock
        return lock


class TeamMappingRequest(BaseModel):
    team_names: dict[int, str] = Field(..., min_length=2)


_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@router.post("/validate-football-video")
async def validate_uploaded_football_video(video: UploadFile = File(...)):
    """Preflight-check a selected video before creating a match upload."""
    filename = video.filename or "validation_video.mp4"
    ext = Path(filename).suffix.lower() or ".mp4"
    content_type = video.content_type or ""
    if content_type and not content_type.startswith("video/") and ext not in _VIDEO_EXTENSIONS:
        return {
            "is_valid": False,
            "status": "invalid",
            "confidence": 0.0,
            "message": "Please upload a video file.",
            "sampled_frames": 0,
            "positive_frame_ratio": 0.0,
            "evidence": {},
            "frame_scores": [],
        }

    validation_dir = Path(settings.UPLOAD_DIR) / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    temp_path = validation_dir / f"{uuid.uuid4().hex}{ext}"

    try:
        with temp_path.open("wb") as outfile:
            while chunk := await video.read(1024 * 1024):
                outfile.write(chunk)
        validation = await asyncio.to_thread(validate_football_video, temp_path)
        return validation.to_dict()
    finally:
        await video.close()
        temp_path.unlink(missing_ok=True)


@router.post("/upload-video")
async def upload_video(
    video: UploadFile = File(...),
    commentary_level: str = Form("Intermediate"),
    commentary_verbosity: str = Form("medium"),
    educational_mode: bool = Form(False),
    commentary_style: str = Form("neutral"),
    football_knowledge: str = Form(""),
    home_team_name: str = Form(""),
    away_team_name: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    """Upload a video and create a match record (pipeline starts after lineup setup)."""
    if commentary_level not in ALLOWED_COMMENTARY_LEVELS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid commentary_level. Expected one of: {', '.join(ALLOWED_COMMENTARY_LEVELS)}",
        )
    if commentary_verbosity not in COMMENTARY_VERBOSITY:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid commentary_verbosity. Expected one of: {', '.join(COMMENTARY_VERBOSITY)}",
        )
    if commentary_style not in COMMENTARY_STYLES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid commentary_style. Expected one of: {', '.join(COMMENTARY_STYLES)}",
        )

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

    validation = await asyncio.to_thread(validate_football_video, video_path)
    if not validation.is_valid:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=validation.message)

    job_service = get_job_service()
    job = job_service.create_job()

    async def upsert_team(name: str) -> Team | None:
        clean_name = name.strip()
        if not clean_name:
            return None
        res = await db.execute(select(Team).where(Team.name == clean_name))
        team = res.scalar_one_or_none()
        if team is None:
            team = Team(name=clean_name)
            db.add(team)
            await db.flush()
        return team

    home_team = await upsert_team(home_team_name)
    away_team = await upsert_team(away_team_name)
    team_name_map = {}
    if home_team:
        team_name_map[0] = home_team.name
    if away_team:
        team_name_map[1] = away_team.name

    match = Match(
        video_path=str(video_path),
        home_team_id=home_team.id if home_team else None,
        away_team_id=away_team.id if away_team else None,
        tracking_job_id=job.job_id,
        tracking_artifacts={
            "commentary_level": commentary_level,
            "commentary_verbosity": commentary_verbosity,
            "educational_mode": educational_mode,
            "commentary_style": commentary_style,
            "football_knowledge": football_knowledge,
            "team_name_map": team_name_map,
            "original_filename": filename,
            "football_video_validation": validation.to_dict(),
        },
        status="team_mapping_pending",
        status_detail="Video uploaded. Waiting for team color detection...",
    )
    db.add(match)
    await db.commit()
    await db.refresh(match)

    return {
        "match_id": match.id,
        "job_id": job.job_id,
        "status": match.status,
        "commentary_level": commentary_level,
        "commentary_verbosity": commentary_verbosity,
        "educational_mode": educational_mode,
        "commentary_style": commentary_style,
        "football_knowledge": football_knowledge,
        "home_team_name": home_team.name if home_team else None,
        "away_team_name": away_team.name if away_team else None,
        "football_video_validation": validation.to_dict(),
        "message": validation.message if validation.status == "uncertain" else "Video uploaded. Proceed to team color mapping.",
    }


@router.post("/match/{match_id}/detect-team-colors")
async def detect_team_colors(match_id: int, db: AsyncSession = Depends(get_db)):
    """Detect anonymous team kit colors before the user assigns real team names."""
    match_lock = await _get_team_color_match_lock(match_id)

    async with match_lock:
        result = await db.execute(select(Match).where(Match.id == match_id))
        match = result.scalar_one_or_none()
        if not match:
            raise HTTPException(status_code=404, detail="Match not found")
        if not match.video_path:
            raise HTTPException(status_code=404, detail="Match has no uploaded video")
        if match.status not in ("team_mapping_pending", "lineup_pending"):
            raise HTTPException(
                status_code=409,
                detail=f"Team color detection is not available while match status is {match.status}.",
            )

        artifacts = dict(match.tracking_artifacts or {})
        existing_colors = artifacts.get("team_colors")
        if isinstance(existing_colors, list) and existing_colors:
            return {"match_id": match.id, "team_colors": existing_colors}

        try:
            async with _TEAM_COLOR_DETECTION_SEMAPHORE:
                team_colors = await asyncio.to_thread(detect_team_colors_preview, match.video_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Team color detection failed: {exc}") from exc

        if len(team_colors) < 2:
            raise HTTPException(
                status_code=422,
                detail="Could not confidently detect two team colors from this video. Try a clearer clip or continue with manual lineup names.",
            )

        artifacts["team_colors"] = team_colors
        match.tracking_artifacts = artifacts
        match.status = "team_mapping_pending"
        match.status_detail = "Team colors detected. Waiting for team name mapping..."
        await db.commit()

        return {"match_id": match.id, "team_colors": team_colors}


@router.post("/match/{match_id}/team-mapping")
async def confirm_team_mapping(
    match_id: int,
    body: TeamMappingRequest,
    db: AsyncSession = Depends(get_db),
):
    """Save user-confirmed mapping from detected color clusters to real teams."""
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    if match.status not in ("team_mapping_pending", "lineup_pending"):
        raise HTTPException(
            status_code=409,
            detail=f"Team mapping cannot be changed while match status is {match.status}.",
        )

    cleaned_names = {
        int(team_id): str(name).strip()
        for team_id, name in body.team_names.items()
        if str(name or "").strip()
    }
    if 0 not in cleaned_names or 1 not in cleaned_names:
        raise HTTPException(status_code=422, detail="Please provide names for both detected teams.")

    async def upsert_team(name: str) -> Team:
        res = await db.execute(select(Team).where(Team.name == name))
        team = res.scalar_one_or_none()
        if team is None:
            team = Team(name=name)
            db.add(team)
            await db.flush()
        return team

    home_team = await upsert_team(cleaned_names[0])
    away_team = await upsert_team(cleaned_names[1])

    artifacts = dict(match.tracking_artifacts or {})
    artifacts["team_name_map"] = {0: home_team.name, 1: away_team.name}
    team_colors = artifacts.get("team_colors")
    if isinstance(team_colors, list):
        for team_color in team_colors:
            if not isinstance(team_color, dict):
                continue
            try:
                team_id = int(team_color.get("team_id"))
            except (TypeError, ValueError):
                continue
            if team_id in cleaned_names:
                team_color["team_name"] = cleaned_names[team_id]
    match.tracking_artifacts = artifacts
    match.home_team_id = home_team.id
    match.away_team_id = away_team.id
    match.status = "lineup_pending"
    match.status_detail = "Team names mapped to detected colors. Waiting for lineup setup..."
    await db.commit()

    return {
        "match_id": match.id,
        "home_team_name": home_team.name,
        "away_team_name": away_team.name,
        "team_colors": artifacts.get("team_colors", []),
        "message": "Team names mapped successfully.",
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

    artifacts = dict(match.tracking_artifacts or {})
    validation = await asyncio.to_thread(validate_football_video, match.video_path)
    artifacts["football_video_validation"] = validation.to_dict()
    match.tracking_artifacts = artifacts
    if not validation.is_valid:
        match.status = "failed"
        match.status_detail = validation.message
        await db.commit()
        raise HTTPException(status_code=422, detail=validation.message)

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
