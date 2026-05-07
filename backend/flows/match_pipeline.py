"""
Match Analysis Pipeline — Prefect flow.

Pipeline topology:
  [tracking_task]          — GPU-heavy: YOLO + Re-ID + event classification
       │
       ├── PARALLEL BLOCK ──────────────────────────────────────────────────────
       │   xT, xG, VAEP, Decision Quality (analytics group — all 4 concurrent)
       │   PbP Commentary              (fires immediately after tracking)
       └─────────────────────────────────────────────────────────────────────
       │   (wait for ALL above)
  [tactical_commentary_task]   — uses analytics outputs + PbP
       │
  [merge_and_tts_task]         — combine PbP + tactical → TTS audio file
       │
  [store_results_task]         — persist to DB + GCS
       │
  [notify_task]                — mark match complete

Run locally (requires Prefect API configured):
    prefect deployment run match-analysis-pipeline/local

Or trigger programmatically:
    from flows.match_pipeline import analyse_match
    asyncio.run(analyse_match(match_id=1))
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from prefect import flow, task, get_run_logger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Individual tasks — each wraps an existing service function
# ---------------------------------------------------------------------------


@task(name="download-video", retries=2, retry_delay_seconds=10)
async def download_video_task(video_url: str, match_id: int) -> str:
    """Download or locate the video file for the given match.

    For GCS URLs (gs://...) the file is streamed to a temp path.
    For local paths the path is returned unchanged.
    """
    run_logger = get_run_logger()
    if video_url.startswith("gs://"):
        from google.cloud import storage as gcs
        import tempfile

        client = gcs.Client()
        bucket_name, blob_path = video_url[5:].split("/", 1)
        blob = client.bucket(bucket_name).blob(blob_path)
        suffix = os.path.splitext(blob_path)[-1] or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        run_logger.info("Downloading gs://%s/%s → %s", bucket_name, blob_path, tmp.name)
        blob.download_to_filename(tmp.name)
        return tmp.name
    # local path
    run_logger.info("Using local video path: %s", video_url)
    return video_url


@task(name="tracking", retries=1, retry_delay_seconds=30, timeout_seconds=7200)
async def tracking_task(video_path: str, match_id: int) -> list[dict]:
    """Run the full tracking pipeline: YOLO detection → Re-ID → event classification.

    Returns the list of raw StatsBomb-format event dicts for the match.
    """
    run_logger = get_run_logger()
    run_logger.info("Starting tracking pipeline for match %d", match_id)

    # Import here to avoid loading GPU models at import time
    from app.services.merged_pipeline_service import process_match_video
    from app.database.database import AsyncSessionLocal
    from app.models.models import Match
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Match).where(Match.id == match_id))
        match = result.scalar_one_or_none()
        if match is None:
            raise ValueError(f"Match {match_id} not found")

        artifacts = match.tracking_artifacts or {}
        await process_match_video(
            match_id=match_id,
            video_path=video_path,
            db=db,
            commentary_level=artifacts.get("commentary_level", "Intermediate"),
            commentary_verbosity=artifacts.get("commentary_verbosity", "medium"),
            educational_mode=artifacts.get("educational_mode", False),
            commentary_style=artifacts.get("commentary_style", "neutral"),
            football_knowledge=artifacts.get("football_knowledge", ""),
            team_name_map=artifacts.get("team_name_map", {}),
        )

    # Fetch stored events for downstream tasks
    from app.services.event_parser import load_match_events
    async with AsyncSessionLocal() as db:
        events = await load_match_events(match_id, db)

    run_logger.info("Tracking complete: %d events extracted", len(events))
    return [e.raw_data for e in events if e.raw_data]


# ── Analytics tasks (run in parallel after tracking) ────────────────────────

@task(name="compute-xt", timeout_seconds=120)
async def compute_xt_task(events: list[dict], match_id: int) -> dict[str, float]:
    """Compute xT (Expected Threat) for all events."""
    from app.analytics.xt import get_xt_value

    xt_map: dict[str, float] = {}
    for ev in events:
        ev_id = str(ev.get("id", ""))
        loc = ev.get("location", [])
        if ev_id and loc and len(loc) >= 2:
            xt_map[ev_id] = get_xt_value(float(loc[0]), float(loc[1]))
    return xt_map


@task(name="compute-xg", timeout_seconds=120)
async def compute_xg_task(events: list[dict], match_id: int) -> dict[str, float]:
    """Compute xG (Expected Goals) for shot events."""
    from app.analytics.xg import compute_xg

    xg_map: dict[str, float] = {}
    for ev in events:
        etype = ev.get("type", {})
        etype_name = etype.get("name", "") if isinstance(etype, dict) else str(etype)
        if etype_name == "Shot":
            ev_id = str(ev.get("id", ""))
            if ev_id:
                xg_map[ev_id] = compute_xg(ev)
    return xg_map


@task(name="compute-vaep", timeout_seconds=300)
async def compute_vaep_task(events: list[dict], match_id: int) -> dict[str, float]:
    """Compute VAEP across the full match event sequence (sequential formula)."""
    from app.analytics.vaep import compute_match_vaep_values

    return compute_match_vaep_values(events)


@task(name="compute-decision-quality", timeout_seconds=600)
async def compute_dq_task(events: list[dict], match_id: int) -> dict:
    """Compute Decision Quality for all players in the match."""
    from app.analytics.decision_quality import compute_match_decision_quality

    return compute_match_decision_quality(events)


# ── Commentary tasks ─────────────────────────────────────────────────────────

@task(name="pbp-commentary", timeout_seconds=1800)
async def pbp_commentary_task(events: list[dict], match_id: int) -> dict:
    """Generate play-by-play commentary for the match.

    Runs in parallel with the analytics group — only needs raw events.
    """
    from app.services.commentary_service import generate_pbp_commentary

    get_run_logger().info("Generating PbP commentary for match %d", match_id)
    return await generate_pbp_commentary(match_id=match_id, events=events)


@task(name="tactical-commentary", timeout_seconds=1800)
async def tactical_commentary_task(
    events: list[dict],
    analytics: dict,
    pbp_output: dict,
    match_id: int,
) -> dict:
    """Generate tactical commentary using analytics outputs + PbP.

    Runs after both the analytics group AND PbP commentary are done.
    """
    from app.services.commentary_service import generate_tactical_commentary

    get_run_logger().info("Generating tactical commentary for match %d", match_id)
    return await generate_tactical_commentary(
        match_id=match_id,
        events=events,
        analytics=analytics,
        pbp_output=pbp_output,
    )


@task(name="merge-tts", timeout_seconds=3600)
async def merge_and_tts_task(
    pbp_output: dict,
    tactical_output: dict,
    match_id: int,
) -> Optional[str]:
    """Merge PbP + tactical commentary and synthesise TTS audio."""
    from app.services.commentary_service import merge_and_synthesise

    get_run_logger().info("Merging commentary and synthesising TTS for match %d", match_id)
    return await merge_and_synthesise(
        match_id=match_id,
        pbp_output=pbp_output,
        tactical_output=tactical_output,
    )


# ── Persistence task ─────────────────────────────────────────────────────────

@task(name="store-results", retries=2, retry_delay_seconds=10)
async def store_results_task(
    match_id: int,
    xt_map: dict,
    xg_map: dict,
    vaep_map: dict,
    dq_result: dict,
    audio_path: Optional[str],
) -> None:
    """Persist all analytics results to the database and GCS."""
    from app.database.database import AsyncSessionLocal
    from app.models.models import Match
    from app.utils.storage import upload_to_gcs
    from sqlalchemy import select

    run_logger = get_run_logger()

    # Upload audio to GCS if available
    audio_gcs_url = None
    if audio_path and os.path.exists(audio_path):
        gcs_key = f"commentary_videos/{match_id}/{os.path.basename(audio_path)}"
        audio_gcs_url = await asyncio.get_event_loop().run_in_executor(
            None, upload_to_gcs, audio_path, gcs_key
        )
        run_logger.info("Commentary audio uploaded: %s", audio_gcs_url)

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Match).where(Match.id == match_id))
        match = result.scalar_one_or_none()
        if match:
            existing = match.tracking_artifacts or {}
            existing.update({
                "prefect_xt_computed": True,
                "prefect_xg_computed": True,
                "prefect_vaep_computed": True,
                "prefect_dq_computed": True,
                "prefect_commentary_audio": audio_gcs_url,
                "dq_players": dq_result.get("players", []),
            })
            match.tracking_artifacts = existing
            match.status = "complete"
            match.status_detail = "All analytics and commentary complete."
            await db.commit()
            run_logger.info("Match %d marked complete", match_id)


@task(name="notify")
async def notify_task(match_id: int) -> None:
    """Log completion — extend to send webhooks or push notifications."""
    get_run_logger().info("Match %d analysis pipeline complete.", match_id)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

@flow(
    name="match-analysis-pipeline",
    description="Full match analysis: tracking → parallel analytics/PbP → tactical commentary → TTS",
    log_prints=True,
)
async def analyse_match(match_id: int, video_url: str) -> dict[str, Any]:
    """Run the complete match analysis pipeline for a given match.

    Args:
        match_id: Database ID of the match record.
        video_url: Local file path or gs://bucket/path to the video.

    Returns:
        Summary dict with keys: xt_map, xg_map, vaep_map, dq_result, audio_path.
    """
    run_logger = get_run_logger()
    run_logger.info("=== Match %d pipeline started ===", match_id)

    # ── Step 0: resolve video ────────────────────────────────────────────────
    video_path = await download_video_task(video_url, match_id)

    # ── Step 1: tracking (sequential GPU — all downstream needs events) ──────
    events = await tracking_task(video_path, match_id)

    if not events:
        run_logger.warning("No events produced for match %d — aborting pipeline", match_id)
        return {}

    # ── Step 2: parallel block — analytics + PbP ─────────────────────────────
    # All five tasks fire simultaneously.
    xt_future   = compute_xt_task.submit(events, match_id)
    xg_future   = compute_xg_task.submit(events, match_id)
    vaep_future = compute_vaep_task.submit(events, match_id)
    dq_future   = compute_dq_task.submit(events, match_id)
    pbp_future  = pbp_commentary_task.submit(events, match_id)

    # Collect results (blocks until all five are done)
    xt_map   = xt_future.result()
    xg_map   = xg_future.result()
    vaep_map = vaep_future.result()
    dq_result = dq_future.result()
    pbp_output = pbp_future.result()

    run_logger.info(
        "Parallel block complete — xT keys: %d, xG keys: %d, VAEP keys: %d, DQ players: %d",
        len(xt_map), len(xg_map), len(vaep_map),
        len(dq_result.get("players", [])),
    )

    # ── Step 3: tactical commentary (needs all analytics + PbP) ─────────────
    analytics = {
        "xt": xt_map,
        "xg": xg_map,
        "vaep": vaep_map,
        "dq": dq_result,
    }
    tactical_output = await tactical_commentary_task(events, analytics, pbp_output, match_id)

    # ── Step 4: merge + TTS ──────────────────────────────────────────────────
    audio_path = await merge_and_tts_task(pbp_output, tactical_output, match_id)

    # ── Step 5: persist ──────────────────────────────────────────────────────
    await store_results_task(match_id, xt_map, xg_map, vaep_map, dq_result, audio_path)

    # ── Step 6: notify ───────────────────────────────────────────────────────
    await notify_task(match_id)

    run_logger.info("=== Match %d pipeline complete ===", match_id)
    return {
        "match_id": match_id,
        "n_events": len(events),
        "n_xt": len(xt_map),
        "n_xg": len(xg_map),
        "n_vaep": len(vaep_map),
        "dq_players": len(dq_result.get("players", [])),
        "audio_path": audio_path,
    }


# ---------------------------------------------------------------------------
# Entry point for local testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    match_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    video_url = sys.argv[2] if len(sys.argv) > 2 else "./uploads/test.mp4"
    result = asyncio.run(analyse_match(match_id=match_id, video_url=video_url))
    print("Pipeline result:", result)
