"""
Robust stable-identity Re-ID system for football tracking.

This module owns the authoritative ByteTrack-ID to stable-identity mapping used by
the live pipeline. Appearance embeddings are provided by the shared ReIDModel,
which prefers FastReID with ViT checkpoints when available, then torchreid, and
finally a hand-crafted fallback.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Set, Tuple
import logging

import numpy as np

from app.event_detection.reid_module import ReIDModel

try:
    from app.event_detection.temporal_aggregator import TemporalIDAggregator

    TEMPORAL_AGGREGATOR_AVAILABLE = True
except ImportError:
    TEMPORAL_AGGREGATOR_AVAILABLE = False
    logging.warning("[Re-ID] temporal_aggregator not found, enhanced temporal aggregation disabled")

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


class StableIdentity:
    """Long-lived identity record used to smooth short-term tracker IDs."""

    def __init__(self, identity_id: int, max_embedding_history: int = 30, mean_window: int = 10):
        self.identity_id = identity_id
        self.detection_ids: Set[int] = set()

        self.embedding_history: deque[np.ndarray] = deque(maxlen=max_embedding_history)
        self.mean_window = max(1, int(mean_window))

        self.position_history: deque[Tuple[float, float]] = deque(maxlen=60)
        self.velocity_history: deque[Tuple[float, float]] = deque(maxlen=20)

        self.kalman_state: Optional[np.ndarray] = None
        self.kalman_P: Optional[np.ndarray] = None

        self.team_id = -1
        self.team_votes: deque[int] = deque(maxlen=10)
        self.jersey_number: Optional[str] = None
        self.jersey_votes: deque[str] = deque(maxlen=12)
        self.jersey_confidences: deque[float] = deque(maxlen=12)
        self.jersey_stability: str = "unknown"

        self.last_seen_frame = 0
        self.first_seen_frame = 0
        self.total_frames_tracked = 0
        self.frames_since_seen = 0
        self.confidence = 1.0
        self.match_scores: deque[float] = deque(maxlen=10)

    @property
    def mean_embedding(self) -> Optional[np.ndarray]:
        if not self.embedding_history:
            return None
        recent = list(self.embedding_history)[-self.mean_window :]
        mean_vector = np.mean(np.stack(recent, axis=0), axis=0)
        return _normalize_vector(mean_vector)

    def infer_jersey_number(self, _crop) -> Optional[str]:
        """Placeholder for future OCR integration."""
        return self.jersey_number

    def add_observation(
        self,
        detection_id: int,
        embedding: Optional[np.ndarray],
        position: Tuple[float, float],
        team_id: int,
        jersey_number: Optional[str],
        jersey_confidence: float,
        frame_idx: int,
    ) -> None:
        self.detection_ids.add(detection_id)

        normalized_embedding = _normalize_vector(embedding)
        if normalized_embedding is not None:
            self.embedding_history.append(normalized_embedding)

        self.position_history.append(position)
        if len(self.position_history) >= 2:
            last = self.position_history[-1]
            prev = self.position_history[-2]
            self.velocity_history.append((last[0] - prev[0], last[1] - prev[1]))

        self._update_kalman(position)

        if team_id >= 0:
            self.team_votes.append(team_id)
            if self.team_votes:
                self.team_id = max(set(self.team_votes), key=self.team_votes.count)

        if jersey_number:
            self.jersey_votes.append(str(jersey_number))
            self.jersey_confidences.append(float(jersey_confidence))
            self.jersey_number = max(set(self.jersey_votes), key=self.jersey_votes.count)
            if len(self.jersey_votes) >= 3 and float(np.mean(self.jersey_confidences)) >= 0.72:
                self.jersey_stability = "stable"
            else:
                self.jersey_stability = "candidate"

        self.last_seen_frame = frame_idx
        self.total_frames_tracked += 1
        self.frames_since_seen = 0

    def _update_kalman(self, measurement: Tuple[float, float]) -> None:
        if self.kalman_state is None:
            self.kalman_state = np.array([measurement[0], measurement[1], 0.0, 0.0], dtype=np.float32)
            self.kalman_P = np.eye(4, dtype=np.float32) * 100.0
            return

        F = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        Q = np.eye(4, dtype=np.float32) * 0.5
        Q[2:, 2:] *= 2.0

        self.kalman_state = F @ self.kalman_state
        self.kalman_P = F @ self.kalman_P @ F.T + Q

        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        R = np.eye(2, dtype=np.float32) * 4.0

        z = np.array(measurement, dtype=np.float32)
        y = z - (H @ self.kalman_state)
        S = H @ self.kalman_P @ H.T + R
        K = self.kalman_P @ H.T @ np.linalg.inv(S)

        self.kalman_state = self.kalman_state + K @ y
        self.kalman_P = (np.eye(4, dtype=np.float32) - K @ H) @ self.kalman_P

    def predict_position(self, frames_ahead: int = 1) -> Tuple[float, float]:
        if self.kalman_state is not None:
            prediction = self.kalman_state.copy()
            F = np.array(
                [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )
            for _ in range(max(1, frames_ahead)):
                prediction = F @ prediction
            return (float(prediction[0]), float(prediction[1]))
        if self.position_history:
            return self.position_history[-1]
        return (0.0, 0.0)

    def compute_appearance_similarity(self, embedding: Optional[np.ndarray], reid_model: ReIDModel) -> float:
        mean_embedding = self.mean_embedding
        if mean_embedding is None or embedding is None:
            return 0.0
        return reid_model.compute_similarity(mean_embedding, embedding)

    def compute_jersey_score(self, jersey_number: Optional[str], jersey_confidence: float, jersey_stability: str) -> tuple[float, str]:
        if not jersey_number or not self.jersey_number:
            return (0.5, "missing")
        if str(jersey_number) == str(self.jersey_number):
            if jersey_stability == "stable" and self.jersey_stability == "stable" and jersey_confidence >= 0.72:
                return (1.0, "stable_match")
            if jersey_confidence >= 0.55:
                return (0.8, "weak_match")
            return (0.65, "candidate_match")
        if jersey_stability == "stable" and self.jersey_stability == "stable" and jersey_confidence >= 0.72:
            return (0.0, "stable_conflict")
        return (0.2, "weak_conflict")


class RobustReIDSystem:
    """
    Stable-ID assignment layer that sits after ByteTrack.

    Detection IDs from the tracker are mapped to long-lived football-player IDs using
    embeddings, motion consistency, team constraints, adaptive thresholds, and temporal
    aggregation.
    """

    def __init__(self, config):
        self.cfg = config
        self.reid_model = ReIDModel(model_path=config.reid.model_path or None, config=config.reid)
        self.backend_status = self.reid_model.get_backend_status()

        self.identities: Dict[int, StableIdentity] = {}
        self.next_identity_id = 1
        self.detection_to_identity: Dict[int, int] = {}

        self.appearance_threshold = getattr(config.reid, "similarity_threshold", config.reid.appearance_threshold)
        self.spatial_threshold = config.reid.spatial_threshold
        self.combined_threshold = config.reid.combined_threshold
        self.embedding_history_size = config.reid.embedding_history_size
        self.max_embedding_history = config.reid.max_embedding_history
        self.embedding_update_interval = max(1, int(getattr(config.reid, "embedding_update_interval", 5)))
        self.use_team_constraint = config.reid.use_team_constraint

        self.id_buffer: Dict[int, deque[int]] = {}
        self.buffer_size = 5

        self.max_identities = 50
        self.max_frames_missing = config.reid.max_lost_frames

        self.use_adaptive = config.reid.use_adaptive_thresholds
        if self.use_adaptive:
            self.velocity_threshold = config.reid.velocity_threshold
            self.high_velocity_appearance_factor = config.reid.high_velocity_appearance_factor
            self.high_velocity_spatial_factor = config.reid.high_velocity_spatial_factor
            self.occlusion_appearance_factor = config.reid.occlusion_appearance_factor
            self.occlusion_spatial_factor = config.reid.occlusion_spatial_factor
            self.occlusion_variance_threshold = config.reid.occlusion_variance_threshold
            logger.info("[Re-ID] Adaptive thresholding: enabled")
        else:
            logger.info("[Re-ID] Adaptive thresholding: disabled")

        self.use_enhanced_temporal = config.reid.use_enhanced_temporal
        self.temporal_aggregator = None
        if self.use_enhanced_temporal and TEMPORAL_AGGREGATOR_AVAILABLE:
            self.temporal_aggregator = TemporalIDAggregator(
                window_size=config.reid.temporal_window_size,
                consensus_threshold=config.reid.temporal_consensus_threshold,
            )
            logger.info(
                "[Re-ID] Enhanced temporal aggregation enabled (window=%s, threshold=%s)",
                config.reid.temporal_window_size,
                config.reid.temporal_consensus_threshold,
            )
        elif self.use_enhanced_temporal:
            logger.warning("[Re-ID] Enhanced temporal aggregation requested but unavailable")
            self.use_enhanced_temporal = False

        self.metrics = None
        self.enable_metrics = config.reid.enable_metrics_tracking

        logger.info(
            "[Re-ID] Initialized backend=%s similarity_threshold=%.3f spatial_threshold=%.1f config=%s weights=%s strict=%s reason=%s",
            self.backend_status["backend"],
            self.appearance_threshold,
            self.spatial_threshold,
            self.backend_status["resolved_config_path"],
            self.backend_status["resolved_weights_path"],
            self.backend_status["strict_fastreid"],
            self.backend_status["fallback_reason"],
        )

    def set_metrics_tracker(self, metrics) -> None:
        self.metrics = metrics
        if metrics and self.enable_metrics:
            logger.info("[Re-ID] Metrics tracking enabled")

    def _compute_scene_context(self, identity: StableIdentity) -> Tuple[float, float]:
        avg_velocity = 0.0
        if identity.velocity_history:
            recent_velocities = list(identity.velocity_history)[-5:]
            magnitudes = [float(np.linalg.norm(v)) for v in recent_velocities]
            avg_velocity = float(np.mean(magnitudes))

        occlusion_level = 0.0
        if len(identity.embedding_history) > 5:
            recent_embeddings = list(identity.embedding_history)[-5:]
            similarities = []
            for idx in range(len(recent_embeddings) - 1):
                similarities.append(self.reid_model.compute_similarity(recent_embeddings[idx], recent_embeddings[idx + 1]))
            if similarities:
                occlusion_level = float(np.std(similarities))

        return avg_velocity, occlusion_level

    def _get_adaptive_thresholds(self, avg_velocity: float, occlusion_level: float) -> Tuple[float, float, str]:
        appearance_threshold = self.appearance_threshold
        spatial_threshold = self.spatial_threshold
        reason = "none"

        high_velocity = avg_velocity > self.velocity_threshold
        high_occlusion = occlusion_level > self.occlusion_variance_threshold

        if high_velocity and high_occlusion:
            appearance_threshold *= min(self.high_velocity_appearance_factor, self.occlusion_appearance_factor)
            spatial_threshold *= max(self.high_velocity_spatial_factor, self.occlusion_spatial_factor)
            reason = "both"
        elif high_velocity:
            appearance_threshold *= self.high_velocity_appearance_factor
            spatial_threshold *= self.high_velocity_spatial_factor
            reason = "velocity"
        elif high_occlusion:
            appearance_threshold *= self.occlusion_appearance_factor
            spatial_threshold *= self.occlusion_spatial_factor
            reason = "occlusion"

        return appearance_threshold, spatial_threshold, reason

    def process_frame(self, frame: np.ndarray, tracks: Dict, frame_idx: int) -> Dict:
        if not tracks:
            self._update_identities(frame_idx)
            self._cleanup_old_identities(frame_idx)
            return {}

        detection_embeddings: Dict[int, Optional[np.ndarray]] = {}
        embedding_track_ids: list[int] = []
        embedding_crops: list[np.ndarray] = []
        for det_id, track in tracks.items():
            if track.is_ball or track.is_referee:
                continue
            should_update_embedding = (
                frame_idx % self.embedding_update_interval == 0
                or getattr(track, "frames_tracked", 0) <= 2
            )
            if should_update_embedding:
                crop = self.reid_model.crop_player(frame, track.bbox)
                if crop is not None:
                    embedding_track_ids.append(det_id)
                    embedding_crops.append(crop)
                detection_embeddings[det_id] = None
            else:
                detection_embeddings[det_id] = None

        if embedding_crops:
            if hasattr(self.reid_model, "extract_embeddings"):
                embeddings = self.reid_model.extract_embeddings(embedding_crops)
            else:
                embeddings = [
                    self.reid_model.extract_embedding(crop)
                    for crop in embedding_crops
                ]
            for det_id, embedding in zip(embedding_track_ids, embeddings):
                detection_embeddings[det_id] = embedding

        stable_tracks = {}
        assigned_identity_ids: Set[int] = set()

        for det_id, track in tracks.items():
            if track.is_ball or track.is_referee:
                stable_tracks[det_id] = track
                continue

            identity_id = self._get_stable_identity(
                detection_id=det_id,
                embedding=detection_embeddings.get(det_id),
                position=track.center,
                team_id=track.team_id,
                jersey_number=getattr(track, "jersey_number", None),
                jersey_confidence=getattr(track, "jersey_confidence", 0.0),
                jersey_stability=getattr(track, "jersey_stability", "unknown"),
                frame_idx=frame_idx,
                occupied_identity_ids=assigned_identity_ids,
            )

            stable_track = track
            stable_track.track_id = identity_id
            stable_tracks[identity_id] = stable_track
            assigned_identity_ids.add(identity_id)

        self._update_identities(frame_idx)
        self._cleanup_old_identities(frame_idx)
        return stable_tracks

    def _get_stable_identity(
        self,
        detection_id: int,
        embedding: Optional[np.ndarray],
        position: Tuple[float, float],
        team_id: int,
        jersey_number: Optional[str],
        jersey_confidence: float,
        jersey_stability: str,
        frame_idx: int,
        occupied_identity_ids: Optional[Set[int]] = None,
    ) -> int:
        occupied_identity_ids = occupied_identity_ids or set()

        if detection_id in self.detection_to_identity:
            identity_id = self.detection_to_identity[detection_id]
            if identity_id in self.identities and identity_id not in occupied_identity_ids:
                self.identities[identity_id].add_observation(
                    detection_id,
                    embedding,
                    position,
                    team_id,
                    jersey_number,
                    jersey_confidence,
                    frame_idx,
                )
                self._update_id_buffer(detection_id, identity_id)
                if self.metrics:
                    self.metrics.record_detection_id(detection_id, identity_id)
                return identity_id

        matched_id = self._match_to_identity(
            embedding=embedding,
            position=position,
            team_id=team_id,
            jersey_number=jersey_number,
            jersey_confidence=jersey_confidence,
            jersey_stability=jersey_stability,
            frame_idx=frame_idx,
            occupied_identity_ids=occupied_identity_ids,
        )

        if matched_id is not None:
            final_id = matched_id
            if self.use_enhanced_temporal and self.temporal_aggregator:
                identity = self.identities[matched_id]
                appearance_score = identity.compute_appearance_similarity(embedding, self.reid_model)
                match_confidence = min(0.95, appearance_score + 0.1) if embedding is not None else 0.7

                self.temporal_aggregator.add_vote(
                    detection_id=detection_id,
                    identity_id=matched_id,
                    confidence=match_confidence,
                    frame_idx=frame_idx,
                )
                stable_id = self.temporal_aggregator.get_stable_id(detection_id)
                if stable_id is not None and stable_id not in occupied_identity_ids:
                    final_id = stable_id
                    if self.metrics:
                        self.metrics.record_consensus_result(achieved=True)
                elif self.metrics:
                    self.metrics.record_consensus_result(achieved=False)

            self.detection_to_identity[detection_id] = final_id
            self.identities[final_id].add_observation(
                detection_id,
                embedding,
                position,
                team_id,
                jersey_number,
                jersey_confidence,
                frame_idx,
            )
            self._update_id_buffer(detection_id, final_id)
            if self.metrics:
                self.metrics.record_detection_id(detection_id, final_id)
            return final_id

        if self.metrics:
            self.metrics.record_match_failure()
        return self._create_new_identity(
            detection_id,
            embedding,
            position,
            team_id,
            jersey_number,
            jersey_confidence,
            jersey_stability,
            frame_idx,
        )

    def _match_to_identity(
        self,
        embedding: Optional[np.ndarray],
        position: Tuple[float, float],
        team_id: int,
        jersey_number: Optional[str],
        jersey_confidence: float,
        jersey_stability: str,
        frame_idx: int,
        occupied_identity_ids: Optional[Set[int]] = None,
    ) -> Optional[int]:
        del frame_idx
        if not self.identities:
            return None

        occupied_identity_ids = occupied_identity_ids or set()
        best_match_id = None
        best_score = -1.0
        best_appearance_score = 0.0
        best_spatial_score = 0.0
        best_motion_score = 0.0
        adapt_reason = "none"

        for identity_id, identity in self.identities.items():
            if identity_id in occupied_identity_ids:
                continue
            if identity.frames_since_seen > self.max_frames_missing:
                continue
            if self.use_team_constraint and team_id >= 0 and identity.team_id >= 0 and team_id != identity.team_id:
                continue

            if self.use_adaptive:
                avg_velocity, occlusion_level = self._compute_scene_context(identity)
                appearance_threshold, spatial_threshold, adapt_reason = self._get_adaptive_thresholds(
                    avg_velocity, occlusion_level
                )
            else:
                appearance_threshold = self.appearance_threshold
                spatial_threshold = self.spatial_threshold
                adapt_reason = "none"

            appearance_score = identity.compute_appearance_similarity(embedding, self.reid_model)
            jersey_score, jersey_status = identity.compute_jersey_score(
                jersey_number, jersey_confidence, jersey_stability
            )
            if jersey_status == "stable_match":
                appearance_threshold *= 0.84
            elif jersey_status in {"weak_match", "candidate_match"}:
                appearance_threshold *= 0.92
            elif jersey_status == "stable_conflict":
                appearance_threshold *= 1.18
            elif jersey_status == "weak_conflict":
                appearance_threshold *= 1.08
            if appearance_score < appearance_threshold:
                continue

            predicted_pos = identity.predict_position(frames_ahead=identity.frames_since_seen + 1)
            distance = float(
                np.sqrt((position[0] - predicted_pos[0]) ** 2 + (position[1] - predicted_pos[1]) ** 2)
            )
            if distance > spatial_threshold:
                continue

            spatial_score = 1.0 - min(1.0, distance / max(spatial_threshold, 1e-8))

            motion_score = 0.5
            if identity.velocity_history:
                avg_velocity_vec = np.mean(np.asarray(list(identity.velocity_history)[-5:], dtype=np.float32), axis=0)
                if float(np.linalg.norm(avg_velocity_vec)) > 1.0 and identity.position_history:
                    last_pos = identity.position_history[-1]
                    actual_disp = (position[0] - last_pos[0], position[1] - last_pos[1])
                    expected_disp = avg_velocity_vec * max(identity.frames_since_seen, 1)
                    disp_diff = float(np.linalg.norm(np.asarray(actual_disp, dtype=np.float32) - expected_disp))
                    motion_score = 1.0 - min(1.0, disp_diff / 100.0)

            combined_score = (
                0.52 * appearance_score
                + 0.20 * spatial_score
                + 0.13 * motion_score
                + 0.15 * jersey_score
            )
            combined_score *= 0.5 + 0.5 * identity.confidence
            if jersey_status == "stable_conflict":
                combined_score *= 0.35
                if appearance_score < min(0.97, self.appearance_threshold + 0.08):
                    continue
            elif jersey_status == "weak_conflict":
                combined_score *= 0.7
            elif jersey_status == "stable_match":
                combined_score *= 1.12
            elif jersey_status == "weak_match":
                combined_score *= 1.05

            if combined_score > best_score:
                best_score = combined_score
                best_match_id = identity_id
                best_appearance_score = appearance_score
                best_spatial_score = spatial_score
                best_motion_score = motion_score

            if adapt_reason != "none":
                logger.debug(
                    "Adaptive thresholds for ID %s: app=%.3f spa=%.1f (reason: %s)",
                    identity_id,
                    appearance_threshold,
                    spatial_threshold,
                    adapt_reason,
                )

        if best_match_id is not None and best_score > self.combined_threshold:
            if self.metrics:
                self.metrics.record_match_score(
                    score=best_score,
                    adaptive_used=(adapt_reason != "none"),
                    identity_id=best_match_id,
                    appearance_score=best_appearance_score,
                    spatial_score=best_spatial_score,
                    motion_score=best_motion_score,
                )
                if adapt_reason != "none":
                    self.metrics.record_adaptive_activation(
                        reason=adapt_reason,
                        appearance_factor=best_appearance_score / max(self.appearance_threshold, 1e-8),
                        spatial_factor=best_spatial_score,
                    )
            return best_match_id

        return None

    def _create_new_identity(
        self,
        detection_id: int,
        embedding: Optional[np.ndarray],
        position: Tuple[float, float],
        team_id: int,
        jersey_number: Optional[str],
        jersey_confidence: float,
        jersey_stability: str,
        frame_idx: int,
    ) -> int:
        del jersey_stability
        new_id = self.next_identity_id
        self.next_identity_id += 1

        identity = StableIdentity(
            new_id,
            max_embedding_history=self.max_embedding_history,
            mean_window=self.embedding_history_size,
        )
        identity.add_observation(
            detection_id,
            embedding,
            position,
            team_id,
            jersey_number,
            jersey_confidence,
            frame_idx,
        )
        identity.first_seen_frame = frame_idx

        self.identities[new_id] = identity
        self.detection_to_identity[detection_id] = new_id
        self._update_id_buffer(detection_id, new_id)

        if self.metrics:
            self.metrics.record_detection_id(detection_id, new_id)

        logger.debug("Created new stable identity %s", new_id)
        return new_id

    def _update_id_buffer(self, detection_id: int, identity_id: int) -> None:
        if detection_id not in self.id_buffer:
            self.id_buffer[detection_id] = deque(maxlen=self.buffer_size)
        self.id_buffer[detection_id].append(identity_id)

    def _update_identities(self, frame_idx: int) -> None:
        for identity in self.identities.values():
            if identity.last_seen_frame < frame_idx:
                identity.frames_since_seen = frame_idx - identity.last_seen_frame
                identity.confidence *= 0.98
            if self.metrics and identity.frames_since_seen == 0:
                self.metrics.record_identity_activity(identity.identity_id)

    def _cleanup_old_identities(self, frame_idx: int) -> None:
        to_remove = []
        for identity_id, identity in self.identities.items():
            if identity.frames_since_seen > self.max_frames_missing * 2:
                to_remove.append(identity_id)

        for identity_id in to_remove:
            del self.identities[identity_id]
            mapped_detection_ids = [
                det_id for det_id, mapped_identity in self.detection_to_identity.items() if mapped_identity == identity_id
            ]
            for det_id in mapped_detection_ids:
                del self.detection_to_identity[det_id]

        if len(self.identities) > self.max_identities:
            least_confident = sorted(self.identities.items(), key=lambda item: item[1].confidence)
            overflow = len(self.identities) - self.max_identities
            for identity_id, _ in least_confident[:overflow]:
                del self.identities[identity_id]

        if self.use_enhanced_temporal and self.temporal_aggregator:
            active_detection_ids = list(self.detection_to_identity.keys())
            self.temporal_aggregator.cleanup(active_detection_ids, frame_idx)

    def get_backend_status(self) -> dict[str, object]:
        return dict(self.backend_status)
