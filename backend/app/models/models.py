"""SQLAlchemy ORM models for the football analysis platform."""

from datetime import datetime, date
from typing import Optional

from sqlalchemy import (
    Integer, String, Float, Text, Date, DateTime, ForeignKey, JSON, UniqueConstraint,
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.database.database import Base


# ── Matches ───────────────────────────────────────────────────────────────


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    home_team_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("teams.id"), nullable=True)
    away_team_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("teams.id"), nullable=True)
    video_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    commentary_video_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    tracking_job_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, unique=True)
    tracking_artifacts: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), default="uploading",
        doc="Pipeline status: uploading | tracking | analytics_processing | commentary_generation | completed | failed",
    )
    status_detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True, doc="Detailed status message")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id], lazy="selectin")
    away_team = relationship("Team", foreign_keys=[away_team_id], lazy="selectin")
    events = relationship("Event", back_populates="match", lazy="selectin")
    lineups = relationship("Lineup", back_populates="match", lazy="selectin")
    player_stats = relationship("PlayerStats", back_populates="match", lazy="selectin")


# ── Teams ─────────────────────────────────────────────────────────────────


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)

    players = relationship("Player", back_populates="team", lazy="selectin")


# ── Players ───────────────────────────────────────────────────────────────


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    team_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("teams.id"), nullable=True)
    position: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    jersey_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    team = relationship("Team", back_populates="players", lazy="selectin")
    stats = relationship("PlayerStats", back_populates="player", lazy="selectin")
    embedding = relationship("PlayerEmbedding", back_populates="player", uselist=False, lazy="selectin")


# ── Events (with embedded freeze frames) ─────────────────────────────────


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False, index=True)
    event_uuid: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    player_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("players.id"), nullable=True)
    team_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("teams.id"), nullable=True)
    period: Mapped[int] = mapped_column(Integer, default=1)
    minute: Mapped[int] = mapped_column(Integer, default=0)
    second: Mapped[int] = mapped_column(Integer, default=0)
    timestamp: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    end_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    end_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    raw_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, doc="Full event JSON including freeze_frame")

    # Relationships
    match = relationship("Match", back_populates="events")
    player = relationship("Player", lazy="selectin")
    team = relationship("Team", lazy="selectin")

    @property
    def freeze_frame(self) -> list[dict]:
        """Extract freeze frame players from raw event data.

        Handles both Component 1 format ({event_frame, players}) and flat list format.
        """
        if self.raw_data and "freeze_frame" in self.raw_data:
            ff = self.raw_data["freeze_frame"]
            if isinstance(ff, dict):
                return ff.get("players", [])
            return ff
        return []

    @property
    def has_freeze_frame(self) -> bool:
        return len(self.freeze_frame) > 0


# ── Player Stats ──────────────────────────────────────────────────────────


class PlayerStats(Base):
    __tablename__ = "player_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"), nullable=False, index=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False, index=True)

    # Core stats
    passes: Mapped[int] = mapped_column(Integer, default=0)
    pass_accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    progressive_passes: Mapped[int] = mapped_column(Integer, default=0)
    carries: Mapped[int] = mapped_column(Integer, default=0)
    shots: Mapped[int] = mapped_column(Integer, default=0)
    touches: Mapped[int] = mapped_column(Integer, default=0)
    pressures: Mapped[int] = mapped_column(Integer, default=0)
    recoveries: Mapped[int] = mapped_column(Integer, default=0)
    progressive_carries: Mapped[int] = mapped_column(Integer, default=0)
    tackles: Mapped[int] = mapped_column(Integer, default=0)
    interceptions: Mapped[int] = mapped_column(Integer, default=0)
    duels_won: Mapped[int] = mapped_column(Integer, default=0)
    duels_total: Mapped[int] = mapped_column(Integer, default=0)

    # Advanced metrics
    xg: Mapped[float] = mapped_column(Float, default=0.0)
    xt: Mapped[float] = mapped_column(Float, default=0.0)
    vaep: Mapped[float] = mapped_column(Float, default=0.0)

    # Rating
    rating: Mapped[float] = mapped_column(Float, default=0.0)

    # Relationships
    player = relationship("Player", back_populates="stats", lazy="selectin")
    match = relationship("Match", back_populates="player_stats", lazy="selectin")


# ── Player Embeddings ─────────────────────────────────────────────────────


class PlayerEmbedding(Base):
    __tablename__ = "player_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"), nullable=False)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    embedding_vector: Mapped[Optional[list]] = mapped_column(JSON, nullable=True, doc="Style embedding vector")
    umap_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    umap_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tsne_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tsne_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    style_cluster: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    __table_args__ = (UniqueConstraint("player_id", "match_id", name="uq_embedding_player_match"),)

    player = relationship("Player", back_populates="embedding", lazy="selectin")


# ── Lineups ───────────────────────────────────────────────────────────────


class Lineup(Base):
    __tablename__ = "lineups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False, index=True)
    team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    formation: Mapped[str] = mapped_column(String(20), nullable=False, doc="e.g. '4-3-3', '4-4-2'")

    match = relationship("Match", back_populates="lineups")
    team = relationship("Team", lazy="selectin")
    players = relationship("LineupPlayer", back_populates="lineup", lazy="selectin", cascade="all, delete-orphan")


class LineupPlayer(Base):
    __tablename__ = "lineup_players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lineup_id: Mapped[int] = mapped_column(Integer, ForeignKey("lineups.id"), nullable=False, index=True)
    player_name: Mapped[str] = mapped_column(String(200), nullable=False)
    jersey_number: Mapped[int] = mapped_column(Integer, nullable=False)
    position_slot: Mapped[int] = mapped_column(Integer, nullable=False, doc="Index in formation layout (0-10)")

    lineup = relationship("Lineup", back_populates="players")
