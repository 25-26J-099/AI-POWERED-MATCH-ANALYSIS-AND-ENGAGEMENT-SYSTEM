"""
Compatibility shim for the legacy gallery-style player Re-ID module.

The authoritative stable-ID mapping now lives in `robust_reid.py`. This module keeps
the old interface intact so existing pipeline calls do not break, while routing any
appearance extraction through the shared ReIDModel backend.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np

from app.event_detection.reid_module import ReIDModel

logger = logging.getLogger(__name__)


def _normalize_vector(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None
    return arr / norm


@dataclass
class PlayerGallery:
    track_id: int
    embedding_history: List[np.ndarray] = field(default_factory=list)
    last_positions: List[Tuple[float, float]] = field(default_factory=list)
    last_velocity: Tuple[float, float] = (0.0, 0.0)
    team_id: int = -1
    lost_frame: int = 0
    predicted_position: Optional[Tuple[float, float]] = None
    jersey_number: Optional[str] = None

    @property
    def mean_embedding(self) -> Optional[np.ndarray]:
        if not self.embedding_history:
            return None
        recent = self.embedding_history[-10:]
        mean_vector = np.mean(np.stack(recent, axis=0), axis=0)
        return _normalize_vector(mean_vector)


class TrajectoryPredictor:
    def predict(self, positions: List[Tuple[float, float]], n_frames_ahead: int = 1) -> Optional[Tuple[float, float]]:
        if len(positions) < 2:
            return positions[-1] if positions else None
        recent = positions[-min(10, len(positions)) :]
        dx = (recent[-1][0] - recent[0][0]) / max(len(recent) - 1, 1)
        dy = (recent[-1][1] - recent[0][1]) / max(len(recent) - 1, 1)
        return (recent[-1][0] + dx * n_frames_ahead, recent[-1][1] + dy * n_frames_ahead)

    def compute_velocity(self, positions: List[Tuple[float, float]]) -> Tuple[float, float]:
        if len(positions) < 2:
            return (0.0, 0.0)
        return (positions[-1][0] - positions[-2][0], positions[-1][1] - positions[-2][1])


class PlayerReIDModule:
    """
    Legacy-compatible gallery module.

    This keeps the historical interface used by the pipeline, but it is no longer the
    source of truth for stable identity assignment. It is kept as a lightweight shim for
    compatibility and future extensions.
    """

    def __init__(self, config):
        self.cfg = config.reid
        self.reid_model = ReIDModel(model_path=self.cfg.model_path or None, config=self.cfg)
        self.trajectory_predictor = TrajectoryPredictor()
        self.gallery: Dict[int, PlayerGallery] = OrderedDict()
        self.max_gallery_size = 100
        self.similarity_threshold = getattr(self.cfg, "similarity_threshold", self.cfg.appearance_threshold)

    def infer_jersey_number(self, _crop) -> Optional[str]:
        """Placeholder for future OCR integration."""
        return None

    def update_gallery(self, frame: np.ndarray, active_tracks: Dict, lost_tracks: Dict, frame_idx: int) -> None:
        for tid, track in active_tracks.items():
            if getattr(track, "is_ball", False) or getattr(track, "is_referee", False):
                continue
            if track.frames_tracked % 5 != 0:
                continue

            crop = self.reid_model.crop_player(frame, track.bbox)
            embedding = self.reid_model.extract_embedding(crop)
            if embedding is None:
                continue

            if tid not in self.gallery:
                self.gallery[tid] = PlayerGallery(track_id=tid, team_id=track.team_id)

            entry = self.gallery[tid]
            entry.embedding_history.append(embedding)
            if len(entry.embedding_history) > self.cfg.max_embedding_history:
                entry.embedding_history = entry.embedding_history[-self.cfg.max_embedding_history :]

            entry.last_positions = list(track.position_history[-20:])
            entry.last_velocity = self.trajectory_predictor.compute_velocity(entry.last_positions)
            entry.team_id = track.team_id
            entry.jersey_number = getattr(track, "jersey_number", None)

        for tid, track in lost_tracks.items():
            if tid not in self.gallery:
                continue
            entry = self.gallery[tid]
            entry.lost_frame = frame_idx
            if track.position_history:
                positions = list(track.position_history)
                entry.predicted_position = self.trajectory_predictor.predict(
                    positions,
                    n_frames_ahead=track.frames_lost,
                )
                entry.last_velocity = self.trajectory_predictor.compute_velocity(positions)

        expired = [
            tid
            for tid, entry in self.gallery.items()
            if entry.lost_frame > 0 and (frame_idx - entry.lost_frame) > self.cfg.max_lost_frames
        ]
        for tid in expired:
            del self.gallery[tid]

        while len(self.gallery) > self.max_gallery_size:
            self.gallery.popitem(last=False)

    def match_new_track(
        self,
        frame: np.ndarray,
        new_bbox: Tuple[float, float, float, float],
        new_team_id: int = -1,
        frame_idx: int = 0,
    ) -> Optional[int]:
        del frame_idx
        crop = self.reid_model.crop_player(frame, new_bbox)
        embedding = self.reid_model.extract_embedding(crop)
        if embedding is None:
            return None

        new_center = ((new_bbox[0] + new_bbox[2]) / 2, (new_bbox[1] + new_bbox[3]) / 2)
        best_match = None
        best_score = -1.0

        for tid, entry in self.gallery.items():
            if entry.lost_frame == 0:
                continue
            if new_team_id >= 0 and entry.team_id >= 0 and new_team_id != entry.team_id:
                continue

            mean_embedding = entry.mean_embedding
            if mean_embedding is None:
                continue

            appearance_score = self.reid_model.compute_similarity(embedding, mean_embedding)
            if appearance_score < self.similarity_threshold:
                continue

            spatial_score = 0.0
            if entry.predicted_position is not None:
                distance = float(
                    np.sqrt(
                        (new_center[0] - entry.predicted_position[0]) ** 2
                        + (new_center[1] - entry.predicted_position[1]) ** 2
                    )
                )
                if distance < self.cfg.spatial_threshold:
                    spatial_score = 1.0 - (distance / max(self.cfg.spatial_threshold, 1e-8))

            score = 0.7 * appearance_score + 0.3 * spatial_score
            if score > best_score:
                best_score = score
                best_match = tid

        if best_match is not None:
            self.gallery[best_match].lost_frame = 0
            logger.debug("[Legacy Re-ID] matched new track to gallery entry %s (score=%.3f)", best_match, best_score)
        return best_match

    def get_gallery_stats(self) -> dict:
        active = sum(1 for entry in self.gallery.values() if entry.lost_frame == 0)
        lost = sum(1 for entry in self.gallery.values() if entry.lost_frame > 0)
        return {
            "gallery_size": len(self.gallery),
            "active_entries": active,
            "lost_entries": lost,
            "backend": self.reid_model.backend,
        }
