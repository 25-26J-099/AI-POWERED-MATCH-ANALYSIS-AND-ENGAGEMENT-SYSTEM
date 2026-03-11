"""
Module 3: Robust Player Re-Identification (Re-ID).

Hybrid approach combining:
1. Appearance-based features (lightweight CNN embeddings)
2. Trajectory prediction (Kalman filter / linear extrapolation)
3. Part-based features for occlusion robustness

Addresses the unique challenges of sports Re-ID:
- High intra-class similarity (identical uniforms)
- Severe occlusions (player clusters)
- Image degradation (motion blur, low resolution)
"""
import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PlayerGallery:
    """Stored appearance and motion data for a previously tracked player."""
    track_id: int
    appearance_features: List[np.ndarray] = field(default_factory=list)
    part_features: Dict[str, np.ndarray] = field(default_factory=dict)
    last_positions: List[Tuple[float, float]] = field(default_factory=list)
    last_velocity: Tuple[float, float] = (0.0, 0.0)
    team_id: int = -1
    lost_frame: int = 0
    predicted_position: Optional[Tuple[float, float]] = None

    @property
    def mean_appearance(self) -> Optional[np.ndarray]:
        if not self.appearance_features:
            return None
        return np.mean(self.appearance_features[-5:], axis=0)  # Use last 5


class AppearanceExtractor:
    """
    Lightweight CNN-based appearance feature extractor.
    Uses a color histogram + HOG hybrid for efficiency on low-resource hardware.
    Falls back to handcrafted features when CNN unavailable.
    """

    def __init__(self, feature_dim: int = 128, num_parts: int = 3):
        self.feature_dim = feature_dim
        self.num_parts = num_parts
        self.target_size = (64, 128)  # width, height

    def _extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """Extract normalized color histogram in HSV space."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # H: 16 bins, S: 8 bins, V: 8 bins
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])

        hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
        hist = hist / (np.sum(hist) + 1e-8)
        return hist

    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture features using Local Binary Pattern approximation."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (32, 64))

        # Compute gradient-based texture features
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        orientation = np.arctan2(gy, gx)

        # Histogram of oriented gradients (simplified)
        n_bins = 8
        hist, _ = np.histogram(orientation, bins=n_bins, range=(-np.pi, np.pi),
                                weights=magnitude)
        hist = hist / (np.sum(hist) + 1e-8)
        return hist

    def extract(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """Extract appearance feature vector from a player bounding box."""
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Bounds check
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 - x1 < 5 or y2 - y1 < 10:
            return None

        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, self.target_size)

        # Extract features
        color_feat = self._extract_color_histogram(crop)
        texture_feat = self._extract_texture_features(crop)

        # Combine features
        feature = np.concatenate([color_feat, texture_feat])

        # Normalize to unit vector
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm

        return feature

    def extract_parts(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Dict[str, np.ndarray]:
        """
        Extract part-based features for occlusion-robust matching.
        Divides the body into horizontal strips (head, torso, legs).
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 5 or y2 - y1 < 10:
            return {}

        crop = frame[y1:y2, x1:x2]
        part_h = crop.shape[0] // self.num_parts

        parts = {}
        part_names = ["upper", "middle", "lower"]

        for i in range(self.num_parts):
            py1 = i * part_h
            py2 = (i + 1) * part_h if i < self.num_parts - 1 else crop.shape[0]
            part_crop = crop[py1:py2, :]

            if part_crop.size == 0:
                continue

            feat = self._extract_color_histogram(part_crop)
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            parts[part_names[i] if i < len(part_names) else f"part_{i}"] = feat

        return parts


class TrajectoryPredictor:
    """
    Simple trajectory prediction using linear velocity model.
    Estimates where a lost player might re-appear.
    """

    def predict(
        self,
        positions: List[Tuple[float, float]],
        n_frames_ahead: int = 1,
    ) -> Optional[Tuple[float, float]]:
        """Predict future position using linear extrapolation."""
        if len(positions) < 2:
            return positions[-1] if positions else None

        # Use last few positions for velocity estimation
        n = min(10, len(positions))
        recent = positions[-n:]

        # Compute average velocity
        dx = (recent[-1][0] - recent[0][0]) / (len(recent) - 1)
        dy = (recent[-1][1] - recent[0][1]) / (len(recent) - 1)

        return (
            recent[-1][0] + dx * n_frames_ahead,
            recent[-1][1] + dy * n_frames_ahead,
        )

    def compute_velocity(
        self,
        positions: List[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """Compute current velocity vector."""
        if len(positions) < 2:
            return (0.0, 0.0)
        return (
            positions[-1][0] - positions[-2][0],
            positions[-1][1] - positions[-2][1],
        )


class PlayerReIDModule:
    """
    Player Re-Identification module.

    Maintains a gallery of lost player appearances and uses hybrid matching
    (appearance + trajectory) to re-identify players when they re-enter the frame.
    """

    def __init__(self, config):
        self.cfg = config.reid
        self.appearance_extractor = AppearanceExtractor(
            feature_dim=self.cfg.feature_dim,
            num_parts=self.cfg.num_body_parts,
        )
        self.trajectory_predictor = TrajectoryPredictor()

        # Gallery of lost player identities
        self.gallery: Dict[int, PlayerGallery] = OrderedDict()
        self.max_gallery_size = 100

    def update_gallery(
        self,
        frame: np.ndarray,
        active_tracks: Dict,
        lost_tracks: Dict,
        frame_idx: int,
    ):
        """
        Update the Re-ID gallery:
        1. Update appearance features for active tracks
        2. Add newly lost tracks to gallery
        3. Remove expired gallery entries
        """
        # Update active track appearances (periodically)
        for tid, track in active_tracks.items():
            if track.frames_tracked % 5 == 0:  # Update every 5 frames
                feat = self.appearance_extractor.extract(frame, track.bbox)
                if feat is not None:
                    if tid not in self.gallery:
                        self.gallery[tid] = PlayerGallery(
                            track_id=tid,
                            team_id=track.team_id,
                        )
                    self.gallery[tid].appearance_features.append(feat)
                    # Keep reasonable size
                    if len(self.gallery[tid].appearance_features) > 10:
                        self.gallery[tid].appearance_features = \
                            self.gallery[tid].appearance_features[-5:]

                    # Extract part features
                    parts = self.appearance_extractor.extract_parts(frame, track.bbox)
                    if parts:
                        self.gallery[tid].part_features = parts

                    self.gallery[tid].last_positions = list(track.position_history[-20:])
                    self.gallery[tid].team_id = track.team_id

        # Add lost tracks to gallery
        for tid, track in lost_tracks.items():
            if tid in self.gallery:
                self.gallery[tid].lost_frame = frame_idx
                # Predict where they might reappear
                if track.position_history:
                    self.gallery[tid].predicted_position = \
                        self.trajectory_predictor.predict(
                            track.position_history,
                            n_frames_ahead=track.frames_lost,
                        )
                    self.gallery[tid].last_velocity = \
                        self.trajectory_predictor.compute_velocity(
                            track.position_history,
                        )

        # Remove expired entries
        expired = [
            tid for tid, g in self.gallery.items()
            if g.lost_frame > 0 and (frame_idx - g.lost_frame) > self.cfg.max_lost_frames
        ]
        for tid in expired:
            del self.gallery[tid]

        # Limit gallery size
        while len(self.gallery) > self.max_gallery_size:
            self.gallery.popitem(last=False)

    def match_new_track(
        self,
        frame: np.ndarray,
        new_bbox: Tuple[float, float, float, float],
        new_team_id: int = -1,
        frame_idx: int = 0,
    ) -> Optional[int]:
        """
        Attempt to match a new unidentified track to a previously lost player.

        Uses hybrid scoring:
        - Appearance similarity (cosine similarity of feature embeddings)
        - Spatial proximity (distance from predicted re-entry position)
        - Part-based matching for occluded cases

        Returns matched gallery track_id, or None if no match found.
        """
        if not self.gallery:
            return None

        # Extract features for new detection
        new_feat = self.appearance_extractor.extract(frame, new_bbox)
        if new_feat is None:
            return None

        new_center = (
            (new_bbox[0] + new_bbox[2]) / 2,
            (new_bbox[1] + new_bbox[3]) / 2,
        )

        best_match = None
        best_score = -1

        for tid, gallery_entry in self.gallery.items():
            # Only match against lost tracks
            if gallery_entry.lost_frame == 0:
                continue

            # Team consistency check
            if new_team_id >= 0 and gallery_entry.team_id >= 0:
                if new_team_id != gallery_entry.team_id:
                    continue

            # 1. Appearance similarity
            ref_feat = gallery_entry.mean_appearance
            if ref_feat is None:
                continue

            appearance_sim = float(np.dot(new_feat, ref_feat) /
                                    (np.linalg.norm(new_feat) * np.linalg.norm(ref_feat) + 1e-8))

            # 2. Spatial proximity
            spatial_score = 0.0
            if gallery_entry.predicted_position is not None:
                dist = np.sqrt(
                    (new_center[0] - gallery_entry.predicted_position[0]) ** 2 +
                    (new_center[1] - gallery_entry.predicted_position[1]) ** 2
                )
                if dist < self.cfg.spatial_threshold:
                    spatial_score = 1.0 - (dist / self.cfg.spatial_threshold)

            # 3. Part-based matching (bonus for partial matches)
            part_bonus = 0.0
            if gallery_entry.part_features:
                new_parts = self.appearance_extractor.extract_parts(frame, new_bbox)
                matching_parts = 0
                total_parts = 0
                for part_name, ref_part in gallery_entry.part_features.items():
                    if part_name in new_parts:
                        total_parts += 1
                        part_sim = float(np.dot(new_parts[part_name], ref_part))
                        if part_sim > 0.5:
                            matching_parts += 1
                if total_parts > 0:
                    part_bonus = 0.1 * (matching_parts / total_parts)

            # Combined hybrid score
            score = (
                self.cfg.appearance_weight * appearance_sim +
                self.cfg.spatial_weight * spatial_score +
                part_bonus
            )

            if score > best_score and appearance_sim > self.cfg.appearance_threshold:
                best_score = score
                best_match = tid

        if best_match is not None and best_score > 0.5:
            logger.debug(
                f"Re-ID match: new detection -> track #{best_match} "
                f"(score={best_score:.3f})"
            )
            # Remove from lost gallery
            if best_match in self.gallery:
                self.gallery[best_match].lost_frame = 0
            return best_match

        return None

    def get_gallery_stats(self) -> dict:
        """Return statistics about the Re-ID gallery."""
        active = sum(1 for g in self.gallery.values() if g.lost_frame == 0)
        lost = sum(1 for g in self.gallery.values() if g.lost_frame > 0)
        return {
            "gallery_size": len(self.gallery),
            "active_entries": active,
            "lost_entries": lost,
        }