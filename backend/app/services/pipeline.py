"""Full analysis pipeline — runs as a background task after upload."""

import traceback
from collections import defaultdict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import async_session
from app.models.models import Match, Event, Player, Team, PlayerStats, PlayerEmbedding
from app.services.event_parser import parse_events
from app.analytics.player_stats import compute_player_stats
from app.analytics.ratings import compute_rating
from app.analytics.embeddings import compute_embeddings


async def _update_status(session: AsyncSession, match_id: int, status: str, detail: str):
    """Update match pipeline status."""
    result = await session.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    if match:
        match.status = status
        match.status_detail = detail
        await session.commit()


async def run_full_pipeline(match_id: int):
    """Execute the complete analysis pipeline.

    Flow:
        1. (Tracking) — handled externally, events are posted to us
        2. Parse & store events
        3. Compute xT, xG, VAEP per event
        4. Aggregate player stats
        5. Compute player embeddings & clusters
        6. Compute player ratings
        7. (Commentary) — triggers external components
        8. Mark complete
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info("🚀 run_full_pipeline started for match_id=%s", match_id)

    async with async_session() as session:
        try:
            # ── Step 1: Retrieve events ──────────────────────────────────
            await _update_status(session, match_id, "analytics_processing", "Preparing event data for analytics...")

            result = await session.execute(
                select(Event).where(Event.match_id == match_id).order_by(Event.period, Event.minute, Event.second)
            )
            events = result.scalars().all()

            logger.info("📊 match_id=%s: Found %s events in DB", match_id, len(events))

            if not events:
                logger.warning("⚠️ match_id=%s: No events found, marking as failed", match_id)
                await _update_status(session, match_id, "failed", "No events found for this match.")
                return

            # ── Step 2: Compute analytics ────────────────────────────────
            await _update_status(
                session,
                match_id,
                "analytics_processing",
                "Computing xT, xG, VAEP, player statistics, and heatmaps...",
            )

            await session.execute(PlayerStats.__table__.delete().where(PlayerStats.match_id == match_id))
            await session.execute(PlayerEmbedding.__table__.delete().where(PlayerEmbedding.match_id == match_id))
            await session.commit()

            # Group events by player
            player_events_map = defaultdict(list)
            events_without_player = 0
            for event in events:
                if event.player_id and event.raw_data:
                    player_events_map[event.player_id].append(event.raw_data)
                elif not event.player_id:
                    events_without_player += 1

            logger.info(
                "📊 match_id=%s: Grouped events into %s players (%s events without player_id)",
                match_id, len(player_events_map), events_without_player,
            )

            if not player_events_map:
                logger.warning(
                    "⚠️ match_id=%s: No events have player_id set — analytics will be empty. "
                    "Check event ingestion (event_parser + pipeline.ingest_events).",
                    match_id,
                )

            # Compute stats for each player
            player_stats_records = []
            players_embedding_data = []

            for player_id, player_raw_events in player_events_map.items():
                logger.info("  → Computing stats for player_id=%s (%s events)", player_id, len(player_raw_events))
                stats = compute_player_stats(player_raw_events)

                rating = compute_rating(
                    vaep=stats["vaep"],
                    xt=stats["xt"],
                    xg=stats["xg"],
                    pass_accuracy=stats["pass_accuracy"],
                    progressive_passes=stats["progressive_passes"],
                    progressive_carries=stats["progressive_carries"],
                    pressures=stats["pressures"],
                    recoveries=stats["recoveries"],
                    tackles=stats["tackles"],
                    interceptions=stats["interceptions"],
                )

                # Upsert player stats
                ps = PlayerStats(
                    player_id=player_id,
                    match_id=match_id,
                    passes=stats["passes"],
                    pass_accuracy=stats["pass_accuracy"],
                    progressive_passes=stats["progressive_passes"],
                    carries=stats["carries"],
                    progressive_carries=stats["progressive_carries"],
                    shots=stats["shots"],
                    touches=stats["touches"],
                    pressures=stats["pressures"],
                    recoveries=stats["recoveries"],
                    tackles=stats["tackles"],
                    interceptions=stats["interceptions"],
                    duels_won=stats["duels_won"],
                    duels_total=stats["duels_total"],
                    xg=stats["xg"],
                    xt=stats["xt"],
                    vaep=stats["vaep"],
                    rating=rating,
                )
                session.add(ps)
                player_stats_records.append(ps)

                # Prepare for embedding computation
                players_embedding_data.append({
                    "player_id": player_id,
                    "player_events": player_raw_events,
                    "xt": stats["xt"],
                    "xg": stats["xg"],
                    "vaep": stats["vaep"],
                    "touches": stats["touches"],
                })

            await session.flush()
            logger.info("📊 match_id=%s: Flushed %s player stats records", match_id, len(player_stats_records))

            # Commit player stats first (safe checkpoint)
            await session.commit()
            logger.info("📊 match_id=%s: Player stats committed to DB", match_id)

            # ── Step 3: Style embeddings ─────────────────────────────────
            await _update_status(
                session,
                match_id,
                "analytics_processing",
                "Computing player style embeddings and ratings...",
            )

            logger.info("📊 match_id=%s: Starting embedding computation for %s players...", match_id, len(players_embedding_data))
            try:
                embedding_results = compute_embeddings(players_embedding_data)
                logger.info("📊 match_id=%s: Embedding computation returned %s results", match_id, len(embedding_results))
                for emb in embedding_results:
                    pe = PlayerEmbedding(
                        player_id=emb["player_id"],
                        match_id=match_id,
                        embedding_vector=emb["embedding"],
                        umap_x=emb["umap_x"],
                        umap_y=emb["umap_y"],
                        tsne_x=emb["tsne_x"],
                        tsne_y=emb["tsne_y"],
                        style_cluster=emb["cluster"],
                    )
                    session.add(pe)
                await session.commit()
                logger.info("📊 match_id=%s: Embeddings committed to DB", match_id)
            except Exception as emb_err:
                logger.warning("⚠️ match_id=%s: Embedding save failed (non-fatal): %s", match_id, emb_err)
                await session.rollback()

            # ── Step 4: Commentary ───────────────────────────────────────
            await _update_status(
                session,
                match_id,
                "commentary_generation",
                "Analytics ready. Waiting for commentary components to consume export data...",
            )
            logger.info("📊 match_id=%s: Commentary stage reached", match_id)

            # Commentary is handled by external components merged via GitHub.
            # Placeholder: once commentary video is ready, update the path.
            # In production, this would call the commentary API and wait for completion.

            # ── Step 5: Done ─────────────────────────────────────────────
            await _update_status(
                session, match_id, "completed",
                f"Analysis complete. Processed {len(events)} events for {len(player_events_map)} players."
            )
            logger.info("✅ match_id=%s: Pipeline complete — %s events, %s players", match_id, len(events), len(player_events_map))

        except Exception as e:
            traceback.print_exc()
            logger.error("❌ match_id=%s: Pipeline error: %s", match_id, e)
            try:
                await session.rollback()
                await _update_status(session, match_id, "failed", f"Pipeline error: {str(e)}")
            except Exception:
                pass


async def ingest_events(match_id: int, raw_events: list[dict]):
    """Receive and store events from Component 1 (tracking + detection).

    Called by the pipeline when events are received.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info("📥 ingest_events: match_id=%s, %s raw events received", match_id, len(raw_events))

    async with async_session() as session:
        events, new_players, new_teams = parse_events(raw_events, match_id)
        logger.info("📥 ingest_events: parsed → %s events, %s new players, %s new teams", len(events), len(new_players), len(new_teams))

        # Add teams first (for FK refs)
        for team in new_teams:
            existing = await session.execute(select(Team).where(Team.name == team.name))
            if not existing.scalar_one_or_none():
                session.add(team)
        await session.flush()

        # Build team name→id map from DB
        team_rows = await session.execute(select(Team))
        team_map = {team.name: team.id for team in team_rows.scalars().all()}

        def _resolve_team_name_from_raw(raw_event: dict) -> str | None:
            raw_team = raw_event.get("team", {})
            if not isinstance(raw_team, dict):
                return None
            team_name = raw_team.get("name")
            if team_name:
                return str(team_name)
            team_id = raw_team.get("id")
            if team_id is not None:
                return f"Team {team_id}"
            return None

        player_team_name_map: dict[str, str] = {}
        ordered_team_names: list[str] = []
        for raw in raw_events:
            resolved_team_name = _resolve_team_name_from_raw(raw)
            if resolved_team_name and resolved_team_name not in ordered_team_names:
                ordered_team_names.append(resolved_team_name)

            p_raw = raw.get("player", {})
            p_name = p_raw.get("name", "") if isinstance(p_raw, dict) else ""
            p_id = p_raw.get("id") if isinstance(p_raw, dict) else None
            player_name = p_name or (f"Player {p_id}" if p_id is not None else "")
            if player_name and resolved_team_name:
                player_team_name_map[player_name] = resolved_team_name

        # Add players and link them to their teams
        for player in new_players:
            existing = await session.execute(select(Player).where(Player.name == player.name))
            existing_player = existing.scalar_one_or_none()
            resolved_team_name = player_team_name_map.get(player.name)
            resolved_team_id = team_map.get(resolved_team_name) if resolved_team_name else None
            if existing_player:
                if resolved_team_id is not None and existing_player.team_id != resolved_team_id:
                    existing_player.team_id = resolved_team_id
                continue

            if resolved_team_id is not None:
                player.team_id = resolved_team_id
            session.add(player)
        await session.flush()

        # Rebuild maps after flush to get DB-assigned IDs
        team_rows = await session.execute(select(Team))
        team_map = {team.name: team.id for team in team_rows.scalars().all()}

        player_rows = await session.execute(select(Player))
        player_map = {player.name: player.id for player in player_rows.scalars().all()}

        logger.info("📥 ingest_events: team_map=%s", team_map)
        logger.info("📥 ingest_events: player_map=%s", player_map)

        # Set match home/away team IDs if not already set
        match_result = await session.execute(select(Match).where(Match.id == match_id))
        match = match_result.scalar_one_or_none()
        if match:
            distinct_team_ids = [team_map[name] for name in ordered_team_names if name in team_map]
            if not distinct_team_ids:
                distinct_team_ids = list(dict.fromkeys(team_map.values()))
            if not match.home_team_id and len(distinct_team_ids) >= 1:
                match.home_team_id = distinct_team_ids[0]
            if not match.away_team_id and len(distinct_team_ids) >= 2:
                match.away_team_id = distinct_team_ids[1]

        # Add events
        for event in events:
            raw_team = (event.raw_data or {}).get("team", {})
            raw_player = (event.raw_data or {}).get("player", {})
            team_name = raw_team.get("name") if isinstance(raw_team, dict) else None
            player_name = raw_player.get("name") if isinstance(raw_player, dict) else None

            # Resolve team: try real name first, then generated temporary name
            resolved_team_id = team_map.get(team_name) if team_name else None
            if resolved_team_id is None and isinstance(raw_team, dict):
                tid = raw_team.get("id")
                if tid:
                    resolved_team_id = team_map.get(f"Team {tid}")
            event.team_id = resolved_team_id

            # Resolve player: try real name first, then generated temporary name
            resolved_player_id = player_map.get(player_name) if player_name else None
            if resolved_player_id is None and isinstance(raw_player, dict):
                pid = raw_player.get("id")
                if pid:
                    # Check both plain and team-prefixed formats
                    resolved_player_id = player_map.get(f"Player {pid}")
                    if resolved_player_id is None:
                        # Try team-prefixed format
                        for pname, p_id in player_map.items():
                            if pname.endswith(f"Player {pid}"):
                                resolved_player_id = p_id
                                break
            event.player_id = resolved_player_id
            session.add(event)
        await session.commit()

        events_with_player = sum(1 for e in events if e.player_id is not None)
        events_with_team = sum(1 for e in events if e.team_id is not None)
        logger.info(
            "📥 ingest_events: committed %s events (%s with player_id, %s with team_id)",
            len(events), events_with_player, events_with_team,
        )

        return len(events)
