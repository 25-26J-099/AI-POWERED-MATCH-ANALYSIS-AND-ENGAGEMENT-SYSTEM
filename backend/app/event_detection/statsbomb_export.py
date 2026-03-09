"""StatsBomb-style event export utilities."""
from __future__ import annotations

import json
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.event_detection.event_detector import GameEvent


class StatsBombExporter:
    """
    Export internal events to a StatsBomb-style JSON events file.

    Notes:
    - Preserves existing internal outputs; this exporter creates an additional file.
    - Supports both GameEvent objects and event dictionaries.
    - Uses StatsBomb-style freeze-frame entries:
      {"location": [x, y], "teammate": bool, "actor": bool, "keeper": bool}
    """

    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        self.frame_width = max(1, int(frame_width))
        self.frame_height = max(1, int(frame_height))
        self.player_registry: Dict[int, Dict[str, Any]] = {}
        self.team_registry: Dict[int, Dict[str, str]] = {
            0: {"name": "Team A"},
            1: {"name": "Team B"},
        }
        self.position_registry: Dict[int, Dict[str, Any]] = {}
        self._possession_counter = 1
        self._last_possession_team: Optional[int] = None

    def set_frame_dimensions(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = max(1, int(frame_width))
        self.frame_height = max(1, int(frame_height))

    def export_to_file(self, events: List[Any], output_path: str) -> str:
        """
        Export events to a StatsBomb-style JSON file.

        The file format follows the typical StatsBomb events payload style:
        {
          "match_id": "...",
          "generated_at": "...",
          "provider": "statsbomb_style",
          "events": [ ... ]
        }
        """
        self._possession_counter = 1
        self._last_possession_team = None

        sb_events = [
            self.convert_event(event, index=i)
            for i, event in enumerate(events)
        ]

        payload = {
            "match_id": str(uuid.uuid4()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provider": "statsbomb_style",
            "events": sb_events,
        }

        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(payload, out, indent=2, ensure_ascii=False)
        return output_path

    def convert_event(self, event: Any, index: int = 0) -> Dict[str, Any]:
        data = self._normalize_event(event)
        timestamp = float(data.get("timestamp") or 0.0)
        team_id = self._safe_int(data.get("team_id"), default=-1)
        player_id = self._safe_int(data.get("player_id"), default=None)
        event_type = str(data.get("type") or "unknown")
        location = self._convert_coordinates(data.get("position"))

        minute = int(timestamp // 60)
        second = int(timestamp % 60)

        # StatsBomb possession is sequence-based, not team-id.
        possession_id = self._update_possession(team_id)

        sb_event: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "index": index,
            "period": self._determine_period(timestamp),
            "timestamp": f"00:{minute:02d}:{second:02d}.000",
            "minute": minute,
            "second": second,
            "type": self._convert_event_type(event_type),
            "possession": possession_id,
            "play_pattern": {"id": 1, "name": "Regular Play"},
            "duration": 0.0,
        }

        if team_id >= 0:
            team_info = self._get_team_info(team_id)
            sb_event["team"] = team_info
            sb_event["possession_team"] = team_info

        if player_id is not None:
            sb_event["player"] = self._get_player_info(player_id)
            sb_event["position"] = self._get_position_info(player_id)

        if location is not None:
            sb_event["location"] = location

        # Event-specific detail containers
        if event_type == "pass":
            sb_event["pass"] = self._create_pass_details()
        elif event_type == "shot":
            sb_event["shot"] = self._create_shot_details()
        elif event_type == "duel":
            sb_event["duel"] = self._create_duel_details()
        elif event_type == "carry":
            sb_event["carry"] = {}
        elif event_type == "clearance":
            sb_event["clearance"] = {"body_part": {"id": 40, "name": "Right Foot"}}
        elif event_type == "interception":
            sb_event["interception"] = {}
        elif event_type == "pressure":
            sb_event["pressure"] = {}
        elif event_type == "foul":
            sb_event["foul_committed"] = {"type": {"id": 1, "name": "Foul"}}

        # Attach source/details as non-breaking metadata extension.
        sb_event["metadata"] = {
            "source": data.get("source"),
            "details": data.get("details"),
        }

        # Preserve freeze-frame structure exactly as produced in events.json.
        freeze_frame = self._preserve_freeze_frame(data.get("freeze_frame"))
        if freeze_frame:
            sb_event["freeze_frame"] = freeze_frame

        return sb_event

    def _normalize_event(self, event: Any) -> Dict[str, Any]:
        if isinstance(event, GameEvent):
            return event.to_dict()
        if isinstance(event, dict):
            return event
        raise TypeError(f"Unsupported event type: {type(event)}")

    def _update_possession(self, team_id: int) -> int:
        if team_id >= 0:
            if self._last_possession_team is None:
                self._last_possession_team = team_id
            elif team_id != self._last_possession_team:
                self._possession_counter += 1
                self._last_possession_team = team_id
        return self._possession_counter

    def _convert_event_type(self, internal_type: str) -> Dict[str, Any]:
        type_mapping = {
            "pass": {"id": 30, "name": "Pass"},
            "shot": {"id": 16, "name": "Shot"},
            "tackle": {"id": 4, "name": "Duel"},
            "possession_change": {"id": 2, "name": "Ball Recovery"},
            "out_of_bounds": {"id": 6, "name": "Ball Out"},
            "sprint": {"id": 3, "name": "Dispossessed"},
            "ball_receipt": {"id": 42, "name": "Ball Receipt*"},
            "carry": {"id": 43, "name": "Carry"},
            "pressure": {"id": 17, "name": "Pressure"},
            "ball_recovery": {"id": 2, "name": "Ball Recovery"},
            "duel": {"id": 4, "name": "Duel"},
            "clearance": {"id": 9, "name": "Clearance"},
            "block": {"id": 6, "name": "Block"},
            "goalkeeper_save": {"id": 23, "name": "Goal Keeper"},
            "goalkeeper_claim": {"id": 23, "name": "Goal Keeper"},
            "miscontrol": {"id": 38, "name": "Miscontrol"},
            "dribble": {"id": 14, "name": "Dribble"},
            "dispossessed": {"id": 3, "name": "Dispossessed"},
            "interception": {"id": 10, "name": "Interception"},
            "dribbled_past": {"id": 39, "name": "Dribbled Past"},
            "foul": {"id": 22, "name": "Foul Committed"},
            "set_piece": {"id": 62, "name": "Free Kick"},
        }
        return type_mapping.get(internal_type, {"id": 0, "name": internal_type})

    def _convert_coordinates(self, pixel_pos: Any) -> Optional[List[float]]:
        if not pixel_pos or not isinstance(pixel_pos, (list, tuple)) or len(pixel_pos) < 2:
            return None
        x_pitch = (float(pixel_pos[0]) / self.frame_width) * 120.0
        y_pitch = (float(pixel_pos[1]) / self.frame_height) * 80.0
        return [round(x_pitch, 1), round(y_pitch, 1)]

    def _preserve_freeze_frame(self, freeze_frame: Any) -> Optional[Dict[str, Any]]:
        """
        Preserve freeze-frame payload exactly as provided by internal event output.
        """
        if not isinstance(freeze_frame, dict):
            return None
        return deepcopy(freeze_frame)

    def _get_team_info(self, team_id: int) -> Dict[str, Any]:
        if team_id in self.team_registry:
            return {"id": team_id, "name": self.team_registry[team_id]["name"]}
        return {"id": team_id, "name": f"Team {team_id}"}

    def _get_player_info(self, player_id: int) -> Dict[str, Any]:
        if player_id in self.player_registry:
            player = self.player_registry[player_id]
            payload = {
                "id": player_id,
                "name": player.get("name", f"Player {player_id}"),
            }
            if "jersey_number" in player:
                payload["jersey_number"] = player["jersey_number"]
            return payload
        return {"id": player_id, "name": f"Player {player_id}"}

    def _get_position_info(self, player_id: int) -> Dict[str, Any]:
        if player_id in self.position_registry:
            return self.position_registry[player_id]
        return {"id": 0, "name": "Unknown"}

    def _create_pass_details(self) -> Dict[str, Any]:
        return {
            "length": 0.0,
            "angle": 0.0,
            "height": {"id": 1, "name": "Ground Pass"},
            "type": {"id": 65, "name": "Open Play"},
        }

    def _create_shot_details(self) -> Dict[str, Any]:
        return {
            "type": {"id": 87, "name": "Open Play"},
            "body_part": {"id": 40, "name": "Right Foot"},
            "technique": {"id": 93, "name": "Normal"},
            "outcome": {"id": 96, "name": "Off T"},
        }

    def _create_duel_details(self) -> Dict[str, Any]:
        return {
            "type": {"id": 11, "name": "Ground Duel"},
            "outcome": {"id": 1, "name": "Success"},
        }

    def _determine_period(self, timestamp: float) -> int:
        return 1 if float(timestamp) < 2700 else 2

    @staticmethod
    def _safe_int(value: Any, default: Optional[int]) -> Optional[int]:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
