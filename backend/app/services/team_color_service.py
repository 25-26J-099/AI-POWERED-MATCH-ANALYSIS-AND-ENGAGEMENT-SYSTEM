"""Quick team-color preview pass for uploaded match videos."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config.pipeline_config import PipelineConfig
from app.event_detection.team_assigner import TeamAssigner
from app.event_detection.tracker import PlayerBallTracker
from app.utils.video_utils import VideoReader

logger = logging.getLogger(__name__)


def detect_team_colors_preview(
    video_path: str,
    max_frames: int = 80,
    frame_skip: int = 10,
) -> list[dict[str, Any]]:
    """
    Run a lightweight Component 1 pass that detects anonymous team kit colors.

    This intentionally stops before full analytics. It samples tracked player torso
    colors, fits the same KMeans team assignment model used by the full pipeline,
    and returns frontend-friendly color metadata for user labeling.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    config = PipelineConfig()
    config.optimization.frame_skip = max(1, int(frame_skip))
    config.optimization.max_input_width = 960
    config.optimization.max_input_height = 540
    config.team_assignment.n_clusters = 2

    tracker = PlayerBallTracker(config)
    team_assigner = TeamAssigner(config)
    reader: VideoReader | None = None

    try:
        tracker.initialize()
        reader = VideoReader(
            video_path,
            frame_skip=config.optimization.frame_skip,
            max_width=config.optimization.max_input_width,
            max_height=config.optimization.max_input_height,
        )

        sampled_tracks = 0
        for frame_idx, frame in reader.read_frames(max_frames=max_frames):
            player_tracks, _ = tracker.process_frame(frame, frame_idx)
            frame_h, frame_w = frame.shape[:2]
            for track in player_tracks.values():
                if track.is_ball or track.is_referee:
                    continue
                cx, cy = track.center
                if not (frame_w * 0.05 < cx < frame_w * 0.95 and frame_h * 0.05 < cy < frame_h * 0.88):
                    continue
                team_assigner.collect_sample(frame, track.bbox)
                sampled_tracks += 1
            if len(team_assigner._color_buffer) >= 40:  # noqa: SLF001 - controlled preview sampling
                break

        if not team_assigner.fit(force=True):
            logger.warning(
                "Team-color preview failed to fit for %s after %s sampled tracks",
                video_path,
                sampled_tracks,
            )
            return []

        return team_assigner.get_team_color_metadata()
    finally:
        if reader is not None:
            reader.release()
