"""Full analysis pipeline — runs as a background task after upload."""

import traceback
from collections import defaultdict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import async_session
from app.models.models import Match, Event, Player, PlayerStats, PlayerEmbedding
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
    async with async_session() as session:
        try:
            # ── Step 1: Retrieve events ──────────────────────────────────
            await _update_status(session, match_id, "detecting", "Processing event data...")

            result = await session.execute(
                select(Event).where(Event.match_id == match_id).order_by(Event.period, Event.minute, Event.second)
            )
            events = result.scalars().all()

            if not events:
                await _update_status(session, match_id, "failed", "No events found for this match.")
                return

            # ── Step 2: Compute analytics ────────────────────────────────
            await _update_status(session, match_id, "analyzing", "Computing xT, xG, VAEP, and player statistics...")

            # Group events by player
            player_events_map = defaultdict(list)
            for event in events:
                if event.player_id and event.raw_data:
                    player_events_map[event.player_id].append(event.raw_data)

            # Compute stats for each player
            player_stats_records = []
            players_embedding_data = []

            for player_id, player_raw_events in player_events_map.items():
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

            # ── Step 3: Style embeddings ─────────────────────────────────
            await _update_status(session, match_id, "analyzing", "Computing player style embeddings...")

            try:
                embedding_results = compute_embeddings(players_embedding_data)
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
            except Exception as emb_err:
                print(f"⚠️  Embedding computation failed (non-fatal): {emb_err}")

            await session.commit()

            # ── Step 4: Commentary ───────────────────────────────────────
            await _update_status(session, match_id, "commentary", "Generating commentary video...")

            # Commentary is handled by external components merged via GitHub.
            # Placeholder: once commentary video is ready, update the path.
            # In production, this would call the commentary API and wait for completion.

            # ── Step 5: Done ─────────────────────────────────────────────
            await _update_status(
                session, match_id, "completed",
                f"Analysis complete. Processed {len(events)} events for {len(player_events_map)} players."
            )

        except Exception as e:
            traceback.print_exc()
            try:
                await _update_status(session, match_id, "failed", f"Pipeline error: {str(e)}")
            except Exception:
                pass


async def ingest_events(match_id: int, raw_events: list[dict]):
    """Receive and store events from Component 1 (tracking + detection).

    Called by the pipeline when events are received.
    """
    async with async_session() as session:
        events, new_players, new_teams = parse_events(raw_events, match_id)

        # Add teams first (for FK refs)
        for team in new_teams:
            existing = await session.execute(select(Player).where(Player.name == team.name))
            if not existing.scalar_one_or_none():
                session.add(team)
        await session.flush()

        # Add players
        for player in new_players:
            existing = await session.execute(select(Player).where(Player.name == player.name))
            if not existing.scalar_one_or_none():
                session.add(player)
        await session.flush()

        # Add events
        for event in events:
            session.add(event)
        await session.commit()

        return len(events)
