"""
Drawing and annotation utilities for video output visualization.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class Annotator:
    """Draws tracking annotations, events, minimap, and HUD on video frames."""

    def __init__(self, config):
        self.cfg = config.visualization
        self.frame_w = 0
        self.frame_h = 0
        self.ball_trail: deque = deque(maxlen=self.cfg.ball_trail_length)
        self.event_display_queue: List[dict] = []
        self.event_display_duration = 90  # frames
        # v4: dynamic team colors (can be overridden after K-means fit)
        self._team_colors = {
            0: self.cfg.team1_color,
            1: self.cfg.team2_color,
        }

    def set_team_colors(self, color_map: Dict[int, Tuple[int, int, int]]):
        """Override team display colors (called after K-means fit)."""
        self._team_colors.update(color_map)

    def _get_team_color(self, team_id: int, is_referee: bool = False) -> Tuple[int, int, int]:
        if is_referee:
            return self.cfg.referee_color
        if team_id in self._team_colors:
            return self._team_colors[team_id]
        # v3: unassigned players get distinct gray color (was lumped with referee)
        return getattr(self.cfg, 'unassigned_color', (180, 180, 180))

    def draw_player(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        track_id: int,
        team_id: int = -1,
        is_referee: bool = False,
    ) -> np.ndarray:
        """Draw player annotation with ellipse and team-colored ID label."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx = (x1 + x2) // 2
        cy_bottom = y2
        w = x2 - x1
        h = y2 - y1

        color = self._get_team_color(team_id, is_referee)

        # Draw ellipse at feet
        axes = (max(10, w // 2), max(8, h // 8))
        cv2.ellipse(frame, (cx, cy_bottom), axes, 0, -45, 235, color, 2)

        # Draw ID label with team-colored background
        label = f"#{track_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        lx = cx - tw // 2
        ly = y1 - 8

        # Background rectangle
        cv2.rectangle(
            frame,
            (lx - 3, ly - th - 3),
            (lx + tw + 3, ly + 3),
            color, -1
        )
        # Text (white on dark, black on bright)
        brightness = sum(color) / 3
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        cv2.putText(frame, label, (lx, ly), font, font_scale, text_color, thickness)

        return frame

    def draw_ball(
        self,
        frame: np.ndarray,
        position: Tuple[int, int],
    ) -> np.ndarray:
        """Draw ball marker as a circle (no trail)."""
        x, y = int(position[0]), int(position[1])
        
        # Draw ball as circle (no trail/direction line)
        radius = 8
        cv2.circle(frame, (x, y), radius, self.cfg.ball_color, -1)  # Filled circle
        cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)  # White border
        
        return frame

    def draw_event(
        self,
        frame: np.ndarray,
        event: dict,
        frame_idx: int,
    ) -> np.ndarray:
        """Draw event notification on frame."""
        self.event_display_queue.append({
            **event,
            "start_frame": frame_idx,
        })

        # Clean expired events
        self.event_display_queue = [
            e for e in self.event_display_queue
            if frame_idx - e["start_frame"] < self.event_display_duration
        ]

        # Draw active events
        y_offset = 60
        for i, evt in enumerate(self.event_display_queue[-3:]):  # Show max 3
            remaining = 1.0 - (frame_idx - evt["start_frame"]) / self.event_display_duration
            alpha = min(1.0, remaining * 2)

            event_type = evt.get("type", "EVENT").upper()
            details = evt.get("details", "")
            text = f">> {event_type}: {details}" if details else f">> {event_type}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

            # Background with transparency effect
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (10, y_offset - th - 10),
                (20 + tw, y_offset + 10),
                self.cfg.event_color, -1
            )
            cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0, frame)

            cv2.putText(
                frame, text,
                (15, y_offset),
                font, font_scale, (255, 255, 255), thickness
            )
            y_offset += th + 25

        return frame

    def draw_possession_bar(
        self,
        frame: np.ndarray,
        team1_pct: float,
        team2_pct: float,
    ) -> np.ndarray:
        """Draw ball possession bar - DISABLED (returns frame unchanged)."""
        # Possession bar disabled per user request
        return frame

    def draw_minimap(
        self,
        frame: np.ndarray,
        player_positions: Dict[int, Tuple[float, float]],
        player_teams: Dict[int, int],
        ball_position: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """Draw a tactical minimap overlay."""
        h, w = frame.shape[:2]
        mw = self.cfg.minimap_width
        mh = self.cfg.minimap_height
        margin = self.cfg.minimap_margin

        # Minimap position (bottom-right)
        mx = w - mw - margin
        my = h - mh - margin

        # Draw minimap background (pitch)
        overlay = frame.copy()
        cv2.rectangle(overlay, (mx, my), (mx + mw, my + mh), (34, 139, 34), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Pitch lines
        cv2.rectangle(frame, (mx + 2, my + 2), (mx + mw - 2, my + mh - 2), (255, 255, 255), 1)
        cv2.line(frame, (mx + mw // 2, my), (mx + mw // 2, my + mh), (255, 255, 255), 1)
        cv2.circle(frame, (mx + mw // 2, my + mh // 2), 20, (255, 255, 255), 1)

        # Goal areas
        cv2.rectangle(frame, (mx + 2, my + mh // 4), (mx + 20, my + 3 * mh // 4), (255, 255, 255), 1)
        cv2.rectangle(frame, (mx + mw - 20, my + mh // 4), (mx + mw - 2, my + 3 * mh // 4), (255, 255, 255), 1)

        # Draw players
        for pid, pos in player_positions.items():
            # Normalize position to minimap
            px = mx + int(pos[0] / w * mw)
            py = my + int(pos[1] / h * mh)
            px = np.clip(px, mx + 3, mx + mw - 3)
            py = np.clip(py, my + 3, my + mh - 3)

            team = player_teams.get(pid, -1)
            color = self._get_team_color(team)
            cv2.circle(frame, (px, py), 4, color, -1)
            cv2.circle(frame, (px, py), 4, (255, 255, 255), 1)

        # Draw ball
        if ball_position is not None:
            bx = mx + int(ball_position[0] / w * mw)
            by = my + int(ball_position[1] / h * mh)
            bx = np.clip(bx, mx + 3, mx + mw - 3)
            by = np.clip(by, my + 3, my + mh - 3)
            cv2.circle(frame, (bx, by), 5, (255, 255, 255), -1)
            cv2.circle(frame, (bx, by), 5, (0, 0, 0), 1)

        return frame

    def draw_stats_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        stats: dict,
    ) -> np.ndarray:
        """Draw frame counter and stats HUD."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Frame info (top-left)
        time_str = f"{frame_idx / fps:.1f}s" if fps > 0 else f"F{frame_idx}"
        cv2.putText(frame, time_str, (10, 25), font, 0.6, (255, 255, 255), 1)

        # Stats (bottom-left)
        y_pos = h - 15
        for key in ["events_total", "players_detected"]:
            if key in stats:
                text = f"{key.replace('_', ' ').title()}: {stats[key]}"
                cv2.putText(frame, text, (10, y_pos), font, 0.45, (200, 200, 200), 1)
                y_pos -= 20

        return frame