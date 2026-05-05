"""Merged orchestration for Component 1 tracking and Component 4 analytics."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config.settings import settings
from app.database.database import async_session
from app.models.models import Match
from app.services.analysis_service import AnalysisRequestOptions
from app.services.dependencies import get_job_service
from app.services.pipeline import ingest_events, run_full_pipeline

logger = logging.getLogger(__name__)


async def _update_match(match_id: int, **updates: Any) -> None:
    async with async_session() as session:
        result = await session.execute(select(Match).where(Match.id == match_id))
        match = result.scalar_one_or_none()
        if match is None:
            return
        for key, value in updates.items():
            if key == "tracking_artifacts" and isinstance(value, dict):
                merged_artifacts = dict(match.tracking_artifacts or {})
                merged_artifacts.update(value)
                setattr(match, key, merged_artifacts)
                continue
            setattr(match, key, value)
        await session.commit()


async def _load_team_names(match_id: int) -> dict[int, str]:
    async with async_session() as session:
        result = await session.execute(
            select(Match)
            .options(selectinload(Match.home_team), selectinload(Match.away_team))
            .where(Match.id == match_id)
        )
        match = result.scalar_one_or_none()
        if match is None:
            return {}
        team_names: dict[int, str] = {}
        if match.home_team and match.home_team.name:
            team_names[0] = match.home_team.name
        if match.away_team and match.away_team.name:
            team_names[1] = match.away_team.name
        artifacts = match.tracking_artifacts or {}
        stored_map = artifacts.get("team_name_map") or {}
        if isinstance(stored_map, dict):
            for key, value in stored_map.items():
                try:
                    team_id = int(key)
                except (TypeError, ValueError):
                    continue
                clean_name = str(value or "").strip()
                if clean_name:
                    team_names[team_id] = clean_name
        return team_names


def _load_statsbomb_events(statsbomb_path: str | Path) -> list[dict]:
    path = Path(statsbomb_path)
    with path.open("r", encoding="utf-8") as infile:
        payload = json.load(infile)
    if isinstance(payload, dict):
        events = payload.get("events", [])
    elif isinstance(payload, list):
        events = payload
    else:
        events = []
    return [event for event in events if isinstance(event, dict)]


async def process_match_video(
    match_id: int,
    input_path: str,
    tracking_job_id: str,
    options: AnalysisRequestOptions | None = None,
) -> None:
    """Run the merged pipeline for a match from upload through analytics."""
    job_service = get_job_service()
    team_names = await _load_team_names(match_id)
    resolved_options = options or AnalysisRequestOptions(
        enable_ml_detector=True,
        ml_model_path=settings.HF_EVENT_DETECTOR_WEIGHTS_FILE,
        team_names=team_names,
    )
    if options is not None and not options.team_names:
        options.team_names = team_names
    logger.info("Merged pipeline starting for match_id=%s tracking_job_id=%s", match_id, tracking_job_id)

    await _update_match(
        match_id,
        status="tracking",
        status_detail="Tracking video and generating StatsBomb events...",
    )

    try:
        job_service.start_job(
            job_id=tracking_job_id,
            input_path=input_path,
            options=resolved_options,
            output_name=f"match_{match_id}_tracking",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Tracking failed to start for match_id=%s tracking_job_id=%s", match_id, tracking_job_id)
        await _update_match(match_id, status="failed", status_detail=f"Tracking failed to start: {exc}")
        return
    await monitor_tracking_job(match_id, tracking_job_id)


async def monitor_tracking_job(match_id: int, tracking_job_id: str) -> None:
    """Wait for an already-started tracking job and hand its outputs to analytics."""
    job_service = get_job_service()
    logger.info("Monitoring tracking job for match_id=%s tracking_job_id=%s", match_id, tracking_job_id)

    await _update_match(
        match_id,
        status="tracking",
        status_detail="Tracking job started. Waiting for StatsBomb event output...",
    )

    while True:
        record = job_service.get_job(tracking_job_id)
        logger.info(
            "Tracking job poll match_id=%s tracking_job_id=%s status=%s",
            match_id,
            tracking_job_id,
            record.status,
        )
        if record.status == "completed":
            break
        if record.status == "failed":
            logger.error(
                "Tracking job failed for match_id=%s tracking_job_id=%s error=%s",
                match_id,
                tracking_job_id,
                record.error,
            )
            await _update_match(
                match_id,
                status="failed",
                status_detail=record.error or "Tracking pipeline failed.",
            )
            return
        await asyncio.sleep(1)

    record = job_service.get_job(tracking_job_id)
    artifact_paths = dict(record.artifact_paths)
    tracking_artifacts = dict(artifact_paths)
    if record.result.get("team_colors") is not None:
        tracking_artifacts["team_colors"] = record.result.get("team_colors", [])
    if record.result.get("team_names") is not None:
        tracking_artifacts["team_names"] = record.result.get("team_names", {})
    statsbomb_path = artifact_paths.get("statsbomb_json") or artifact_paths.get("json")
    logger.info(
        "Tracking job completed for match_id=%s tracking_job_id=%s statsbomb_path=%s",
        match_id,
        tracking_job_id,
        statsbomb_path,
    )

    if not statsbomb_path:
        logger.error("No StatsBomb artifact found for match_id=%s tracking_job_id=%s", match_id, tracking_job_id)
        await _update_match(
            match_id,
            status="failed",
            status_detail="Tracking completed but no StatsBomb event artifact was produced.",
            tracking_artifacts=tracking_artifacts,
        )
        return

    try:
        raw_events = _load_statsbomb_events(statsbomb_path)
        logger.info(
            "Loaded %s StatsBomb events for match_id=%s tracking_job_id=%s",
            len(raw_events),
            match_id,
            tracking_job_id,
        )
        await _update_match(
            match_id,
            status="analytics_processing",
            status_detail="StatsBomb events received. Launching Component 4 analytics...",
            tracking_artifacts=tracking_artifacts,
        )
        await ingest_events(match_id, raw_events)
        logger.info("Event ingestion complete for match_id=%s", match_id)
        await run_full_pipeline(match_id)
        logger.info("Component 4 analytics pipeline complete for match_id=%s", match_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Merged pipeline failed after tracking for match_id=%s", match_id)
        await _update_match(match_id, status="failed", status_detail=f"Merged pipeline failed: {exc}")
