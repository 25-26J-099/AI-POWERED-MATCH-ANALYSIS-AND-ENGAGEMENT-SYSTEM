"""
Team Assignment Module (v4).

v4 fixes from real test results:
- Referee classification is now vote-based, NEVER permanent
- "Far from both teams" referee check removed (too aggressive)
- Team assignment is sticky: once assigned, requires 70% vote to override
- Overlay color auto-mapping from HSV cluster centers
- Better handling of color extraction failures (keep previous assignment)
"""
import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List, Any
from sklearn.cluster import KMeans
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


class TeamAssigner:
    def __init__(self, config):
        self.cfg = config.team_assignment
        self.team_colors: Dict[int, np.ndarray] = {}
        self.kmeans: Optional[KMeans] = None
        self.is_fitted = False

        # Team assignment: majority-vote history per track
        self._vote_history: Dict[int, List[int]] = defaultdict(list)
        self._stable: Dict[int, int] = {}
        self._vote_window = getattr(self.cfg, 'vote_window', 25)
        self._vote_threshold = getattr(self.cfg, 'vote_threshold', 0.55)

        # Referee: vote-based, NOT permanent
        self._ref_votes: Dict[int, List[bool]] = defaultdict(list)
        self._ref_window = 20

        # Color samples for initial fit
        self._color_buffer: List[np.ndarray] = []
        self._min_samples = 20

    def extract_jersey_color(
        self, frame: np.ndarray, bbox: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """
        Extract dominant jersey color from the torso region, filtering out
        pitch-green pixels that would otherwise dominate the color estimate.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = y2 - y1, x2 - x1
        if h < 10 or w < 5:
            return None
        fh, fw = frame.shape[:2]
        top = max(0, min(y1 + int(h * self.cfg.jersey_region_top), fh - 1))
        bot = max(top + 1, min(y1 + int(h * self.cfg.jersey_region_bottom), fh))
        left = max(0, min(x1 + int(w * self.cfg.jersey_region_left), fw - 1))
        right = max(left + 1, min(x1 + int(w * self.cfg.jersey_region_right), fw))
        crop = frame[top:bot, left:right]
        if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)
        if len(pixels) < 5:
            return None

        # v4: Filter out pitch-green pixels BEFORE clustering
        # Pitch green typically: H in 35-80, S>25 (avoid catching blue-green)
        h_vals = pixels[:, 0]
        s_vals = pixels[:, 1]
        is_green = (h_vals >= 35) & (h_vals <= 80) & (s_vals >= 25)
        non_green = pixels[~is_green]

        # Use non-green pixels if enough remain (>30% of crop)
        if len(non_green) >= max(5, len(pixels) * 0.15):
            use_pixels = non_green
        else:
            # Fallback: use all pixels but pick NON-dominant cluster
            use_pixels = pixels

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        n_clusters = 2 if len(use_pixels) >= 10 else 1
        try:
            _, labels, centers = cv2.kmeans(
                use_pixels, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )
        except Exception:
            return None

        if n_clusters == 1:
            return centers[0]

        counts = Counter(labels.ravel())
        dominant_idx = counts.most_common(1)[0][0]

        # v4: If we used green-filtered pixels, take dominant (it's the jersey)
        if len(non_green) >= max(5, len(pixels) * 0.15):
            return centers[dominant_idx]

        # If we used all pixels (green filter removed too much),
        # pick the cluster FURTHEST from pitch green (H~60, S~80, V~140)
        pitch_ref = np.array([60.0, 80.0, 140.0])
        dists = [np.linalg.norm(centers[i] - pitch_ref) for i in range(n_clusters)]
        best_idx = int(np.argmax(dists))
        return centers[best_idx]

    def collect_sample(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]):
        color = self.extract_jersey_color(frame, bbox)
        if color is not None:
            self._color_buffer.append(color)

    def fit(self, force: bool = False) -> bool:
        if not force and len(self._color_buffer) < self._min_samples:
            return False
        if not self._color_buffer:
            return False
        features = np.array(self._color_buffer)
        try:
            self.kmeans = KMeans(n_clusters=self.cfg.n_clusters, random_state=42, n_init=10)
            self.kmeans.fit(features)
            self.is_fitted = True
            for i, c in enumerate(self.kmeans.cluster_centers_):
                self.team_colors[i] = c
                logger.info(f"Team {i} color center (HSV): {c}")
            logger.info(f"Team assignment fitted with {len(features)} samples")
            return True
        except Exception as e:
            logger.warning(f"K-means fitting failed: {e}")
            return False

    def assign_team(
        self, frame: np.ndarray, bbox: Tuple[float, float, float, float], track_id: int,
    ) -> int:
        """Assign team with sticky majority voting. Falls back to previous assignment."""
        if not self.is_fitted:
            return self._stable.get(track_id, -1)

        color = self.extract_jersey_color(frame, bbox)
        if color is None:
            # v4: ALWAYS return previous stable assignment, never -1 for existing tracks
            return self._stable.get(track_id, -1)

        try:
            raw = int(self.kmeans.predict(color.reshape(1, -1))[0])
        except Exception:
            return self._stable.get(track_id, -1)

        # Append vote and trim
        self._vote_history[track_id].append(raw)
        if len(self._vote_history[track_id]) > self._vote_window:
            self._vote_history[track_id] = self._vote_history[track_id][-self._vote_window:]

        hist = self._vote_history[track_id]

        # v4: If we already have a stable assignment, require strong evidence to change
        current_stable = self._stable.get(track_id, -1)
        if current_stable >= 0 and len(hist) >= 3:
            # Count votes for the DIFFERENT team
            other_count = sum(1 for v in hist if v != current_stable)
            # Only switch if >70% of recent votes disagree
            if other_count > len(hist) * 0.7:
                winner = 1 - current_stable  # flip
                self._stable[track_id] = winner
                return winner
            return current_stable

        # New track: assign with lower threshold
        if len(hist) >= 2:
            winner, cnt = Counter(hist).most_common(1)[0]
            if cnt > len(hist) * 0.4:
                self._stable[track_id] = winner
                return winner

        # Not enough votes — use raw if no stable
        if current_stable >= 0:
            return current_stable
        self._stable[track_id] = raw
        return raw

    def classify_referee(
        self, frame: np.ndarray, bbox: Tuple[float, float, float, float],
        track_id: int,
    ) -> bool:
        """
        Vote-based referee classification. Returns True only if
        a clear majority of recent checks say 'referee'.
        
        v4: NO permanent flags. Each call is one vote.
        Removed 'far from both teams' check (too aggressive).
        """
        color = self.extract_jersey_color(frame, bbox)
        is_ref = False

        if color is not None:
            h, s, v = color
            # Very dark kit (black referee uniform)
            if v < 50 and s < 50:
                is_ref = True
            # Bright yellow/green referee vest (very specific range)
            elif 20 <= h <= 35 and s > 180 and v > 160:
                is_ref = True

        # v4: Accumulate votes, require strong majority
        self._ref_votes[track_id].append(is_ref)
        if len(self._ref_votes[track_id]) > self._ref_window:
            self._ref_votes[track_id] = self._ref_votes[track_id][-self._ref_window:]

        votes = self._ref_votes[track_id]
        if len(votes) >= 5:
            ref_pct = sum(votes) / len(votes)
            return ref_pct > 0.7  # 70% of recent checks must say referee
        return False

    def get_team_display_colors(self):
        """
        Return BGR display colors that match the actual jersey HSV clusters.
        Maps each cluster's HSV center to a representative BGR color.
        """
        if not self.is_fitted:
            return {0: (0, 0, 220), 1: (220, 0, 0)}

        colors = {}
        for team_id, hsv_center in self.team_colors.items():
            h, s, v = hsv_center
            # Create a small HSV pixel and convert to BGR
            hsv_pixel = np.uint8([[[int(h), int(s), int(v)]]])
            bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
            b, g, r = int(bgr_pixel[0, 0, 0]), int(bgr_pixel[0, 0, 1]), int(bgr_pixel[0, 0, 2])
            # Boost saturation for visibility
            max_c = max(b, g, r, 1)
            scale = min(255 / max_c, 1.5)
            colors[team_id] = (
                min(255, int(b * scale)),
                min(255, int(g * scale)),
                min(255, int(r * scale)),
            )
        return colors

    def get_team_color_metadata(self, team_names: Optional[Dict[int, str]] = None) -> List[Dict[str, Any]]:
        """
        Return frontend/API-friendly color metadata for each detected team cluster.

        Internal team ids remain zero-based for the tracking pipeline. The display
        label is one-based because users naturally read the detected clusters as
        Team 1 and Team 2.
        """
        if not self.is_fitted:
            return []

        team_names = team_names or {}
        display_colors = self.get_team_display_colors()
        metadata: List[Dict[str, Any]] = []
        for team_id in sorted(self.team_colors.keys()):
            hsv_center = self.team_colors[team_id]
            bgr = display_colors.get(team_id)
            if bgr is None:
                continue
            b, g, r = [int(v) for v in bgr]
            rgb = {"r": r, "g": g, "b": b}
            metadata.append({
                "team_id": int(team_id),
                "detected_label": f"Team {int(team_id) + 1}",
                "team_name": team_names.get(int(team_id)) or f"Team {int(team_id) + 1}",
                "color_name": self._name_from_hsv(hsv_center),
                "hex": f"#{r:02X}{g:02X}{b:02X}",
                "rgb": rgb,
                "bgr": {"b": b, "g": g, "r": r},
                "hsv": {
                    "h": round(float(hsv_center[0]), 2),
                    "s": round(float(hsv_center[1]), 2),
                    "v": round(float(hsv_center[2]), 2),
                },
            })
        return metadata

    @staticmethod
    def _name_from_hsv(hsv_center: np.ndarray) -> str:
        h, s, v = [float(x) for x in hsv_center]
        if v < 55:
            return "Black"
        if s < 35:
            return "White" if v > 185 else "Gray"

        if h < 8 or h >= 170:
            return "Red"
        if h < 24:
            return "Orange"
        if h < 35:
            return "Yellow"
        if h < 86:
            return "Green"
        if h < 100:
            return "Cyan"
        if h < 131:
            return "Blue"
        if h < 151:
            return "Purple"
        return "Pink"

    def reset_track(self, track_id: int):
        self._vote_history.pop(track_id, None)
        self._stable.pop(track_id, None)
        self._ref_votes.pop(track_id, None)
