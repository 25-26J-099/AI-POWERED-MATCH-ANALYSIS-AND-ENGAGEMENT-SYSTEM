import asyncio
import os
import shutil
import logging
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.database.database import async_session
from app.models.models import Match, Event, PlayerStats, CommentaryOutput
from app.services.commentary_parser import parse_combined_events
import app.commentary.merged as merged
import app.commentary.tac_commentary as tac
import app.commentary.pbp_commentary as pbp
from app.config.settings import settings

logger = logging.getLogger(__name__)
ALLOWED_COMMENTARY_LEVELS = {"Beginner", "Intermediate", "Expert"}


def _player_stat_vaep_total(stat: PlayerStats) -> float:
    offensive_vaep = getattr(stat, "offensive_vaep", None)
    defensive_vaep = getattr(stat, "defensive_vaep", None)
    if offensive_vaep is not None or defensive_vaep is not None:
        return float(offensive_vaep or 0.0) + float(defensive_vaep or 0.0)
    return float(getattr(stat, "vaep", 0.0) or 0.0)


def _player_stat_xg_total(stat: PlayerStats) -> float:
    return float(getattr(stat, "xg_total", getattr(stat, "xg", 0.0)) or 0.0)


def _player_stat_xt_total(stat: PlayerStats) -> float:
    return float(getattr(stat, "xt_total", getattr(stat, "xt", 0.0)) or 0.0)


def _resolve_tactical_commentary_level(match: Match) -> str:
    stored_level = None
    if isinstance(match.tracking_artifacts, dict):
        stored_level = match.tracking_artifacts.get("commentary_level")
    if stored_level in ALLOWED_COMMENTARY_LEVELS:
        return str(stored_level)

    settings_level = getattr(settings, "COMMENTARY_LEVEL", "Intermediate")
    if settings_level in ALLOWED_COMMENTARY_LEVELS:
        return str(settings_level)
    return "Intermediate"


async def generate_commentary(match_id: int):
    """
    Executes the full commentary pipeline asynchronously for a given match.
    Retrieves events and analytics, builds context, runs the combined
    tactical and play-by-play threads, generates TTS, composites the video,
    and updates the database.
    """
    async with async_session() as session:
        # Fetch match
        result = await session.execute(select(Match).where(Match.id == match_id))
        match = result.scalar_one_or_none()
        if not match:
            logger.error(f"Commentary failed: match {match_id} not found.")
            return

        match.status = "play_by_play_commentary"
        match.status_detail = "Generating commentary timelines"
        await session.commit()

        # Fetch events natively
        events_result = await session.execute(
            select(Event).where(Event.match_id == match_id).order_by(Event.timestamp)
        )
        events = events_result.scalars().all()
        
        # Build combined JSON structures
        combined_events = [e.raw_data for e in events if e.raw_data]
        if not combined_events:
            logger.warning(f"Commentary skipped: no raw event data found for match {match_id}.")
            return
            
        event_json, threesixty_json = parse_combined_events(combined_events)
        
        # Build threesixty lookup
        threesixty_lookup = {}
        for item in threesixty_json:
            uid = item.get("event_uuid")
            if uid:
                threesixty_lookup[uid] = item
                
        # Build Analytics Context for LLM Prompts
        stats_result = await session.execute(
            select(PlayerStats).options(selectinload(PlayerStats.player)).where(PlayerStats.match_id == match_id)
        )
        stats = stats_result.scalars().all()
        
        # Sort top players by total value
        stats_sorted = sorted(stats, key=_player_stat_vaep_total, reverse=True)
        top_players = [
            f"{s.player.name} (VAEP: {_player_stat_vaep_total(s):.2f}, xG: {_player_stat_xg_total(s):.2f}, xT: {_player_stat_xt_total(s):.2f})"
            for s in stats_sorted[:5]
            if s.player
        ]
        
        if top_players:
            analytics_context = "Top Performers:\n" + "\n".join(f"- {p}" for p in top_players)
        else:
            analytics_context = "No specific match analytics available."
            
        logger.info(f"Analytics context passed to commentary:\n{analytics_context}")

        # Update status before heavy processing
        match.status = "expert_commentary"
        match.status_detail = "Processing tactical insights and play-by-play"
        await session.commit()

        tactical_level = _resolve_tactical_commentary_level(match)
        logger.info("Commentary pipeline using tactical level=%s for match_id=%s", tactical_level, match_id)
        
        # Determine source video (fallback to demo if testing, else use uploaded video)
        video_path = "D:\\ResearchPoject\\AI-POWERED-MATCH-ANALYSIS-AND-ENGAGEMENT-SYSTEM\\backend\\app\\commentary\\Demovid.mp4"
        if match.video_path and os.path.exists(match.video_path):
             video_path = match.video_path

        clip_filename = None
        if isinstance(match.tracking_artifacts, dict):
            clip_filename = match.tracking_artifacts.get("original_filename")
        if not clip_filename:
            clip_filename = os.path.basename(video_path)
        
        # Extract commentary logic to sync thread
        def run_sync_pipeline():
            # Patch paths to make models work regardless of CWD
            base_dir = os.path.dirname(os.path.abspath(tac.__file__))
            tac.TAC_MODELFILE = os.path.join(base_dir, "Tacticalmodel", "Modelfile")
            pbp.PBP_MODELFILE = os.path.join(base_dir, "PbpModel", "Modelfile")
            
            # Point to local ollama endpoint if configured
            ollama_url = getattr(settings, "OLLAMA_URL", "http://localhost:11434")
            tac.OLLAMA_URL = f"{ollama_url}/api/chat"
            pbp.OLLAMA_URL = f"{ollama_url}/api/chat"
            pbp.OLLAMA_GENERATE_URL = f"{ollama_url}/api/generate"
            pbp.OLLAMA_TAGS_URL = f"{ollama_url}/api/tags"
            
            # Monkeypatch progress to update state internally if necessary
            def progress_cb(msg):
                logger.info(f"Commentary Pipeline: {msg}")
                
            out_file = None
            def done_cb(output_path, error=None):
                nonlocal out_file
                if error:
                    logger.error(f"Commentary Pipeline Error: {error}")
                out_file = output_path
            
            merged.run_pipeline(
                events=event_json,
                threesixty_lookup=threesixty_lookup,
                video_file=video_path,
                clip_filename=clip_filename,
                level=tactical_level,
                analytics_context=analytics_context,
                progress_callback=progress_cb,
                done_callback=done_cb
            )
            return out_file

        # Run heavy ML logic in a thread
        loop = asyncio.get_event_loop()
        final_video = await loop.run_in_executor(None, run_sync_pipeline)
        
        if not final_video or not os.path.exists(final_video):
            match.status = "failed"
            match.status_detail = "Commentary video generation failed"
            await session.commit()
            return
            
        # Move generated video to absolute destination folder
        dest_dir = getattr(settings, "COMMENTARY_DIR", "./commentary_videos")
        os.makedirs(dest_dir, exist_ok=True)
        final_dest = os.path.join(dest_dir, f"match_{match_id}_commentary.mp4")
        shutil.move(final_video, final_dest)
        
        # Store metadata
        match.commentary_video_path = final_dest
        
        # Create output record
        out_record = CommentaryOutput(
            match_id=match_id,
            commentary_type="merged",
            commentary_text="Merged tactical and play-by-play commentary generated.",
            video_path=final_dest
        )
        session.add(out_record)
        
        # Complete
        match.status = "completed"
        match.status_detail = "Analysis and commentary complete"
        await session.commit()
        logger.info(f"🎉 Commentary pipeline for match {match_id} completed successfully.")
