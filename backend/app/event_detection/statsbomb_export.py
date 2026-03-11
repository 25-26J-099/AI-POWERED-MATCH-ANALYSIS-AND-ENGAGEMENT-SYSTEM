"""StatsBomb-style event export utilities."""
from __future__ import annotations

import json
import math
import re
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.event_detection.event_detector import GameEvent


_PASS_DETAILS_RE = re.compile(r"Pass\s+#(?P<sender>\d+)->(?P<recipient>\d+)\s+\(d=(?P<distance>[\d.]+)px\)")
_CARRY_DETAILS_RE = re.compile(r"Carry\s+\(dist=(?P<distance>[\d.]+)px\)")


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
        self._last_player_locations: Dict[int, List[float]] = {}

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
        self._last_player_locations = {}

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
        default_location = self._convert_coordinates(data.get("position"))
        freeze_frame = self._preserve_freeze_frame(data.get("freeze_frame"))
        start_location, end_location, recipient_id = self._infer_action_locations(
            event_type=event_type,
            player_id=player_id,
            team_id=team_id,
            data=data,
            freeze_frame=freeze_frame,
            default_location=default_location,
        )
        location = start_location or default_location

        total_seconds = max(0, int(timestamp))
        hour = total_seconds // 3600
        minute = (total_seconds % 3600) // 60
        second = total_seconds % 60

        # StatsBomb possession is sequence-based, not team-id.
        possession_id = self._update_possession(team_id)

        sb_event: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "index": index,
            "period": self._determine_period(timestamp),
            "timestamp": f"{hour:02d}:{minute:02d}:{second:02d}.000",
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
            sb_event["pass"] = self._create_pass_details(
                start_location=start_location,
                end_location=end_location,
                recipient_id=recipient_id,
            )
        elif event_type == "shot":
            sb_event["shot"] = self._create_shot_details()
        elif event_type == "duel":
            sb_event["duel"] = self._create_duel_details()
        elif event_type == "carry":
            sb_event["carry"] = self._create_carry_details(end_location=end_location)
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
        if freeze_frame:
            sb_event["freeze_frame"] = freeze_frame

        self._remember_player_location(
            player_id=player_id,
            current_location=self._freeze_frame_player_location(freeze_frame, player_id),
            fallback_location=end_location or location,
        )

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

    def _infer_action_locations(
        self,
        event_type: str,
        player_id: Optional[int],
        team_id: int,
        data: Dict[str, Any],
        freeze_frame: Optional[Dict[str, Any]],
        default_location: Optional[List[float]],
    ) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[int]]:
        if event_type == "pass":
            return self._infer_pass_locations(player_id, team_id, data, freeze_frame, default_location)
        if event_type == "carry":
            return self._infer_carry_locations(player_id, team_id, data, freeze_frame, default_location)
        return default_location, None, None

    def _infer_pass_locations(
        self,
        player_id: Optional[int],
        team_id: int,
        data: Dict[str, Any],
        freeze_frame: Optional[Dict[str, Any]],
        default_location: Optional[List[float]],
    ) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[int]]:
        details = str(data.get("details") or "")
        parsed = _PASS_DETAILS_RE.search(details)
        recipient_id = self._safe_int(parsed.group("recipient"), None) if parsed else None
        pass_distance_px = float(parsed.group("distance")) if parsed else None

        current_actor_location = self._freeze_frame_player_location(freeze_frame, player_id)
        recipient_location = self._freeze_frame_player_location(freeze_frame, recipient_id)
        ball_location = self._freeze_frame_ball_location(freeze_frame)
        start_location = self._last_player_locations.get(player_id) or current_actor_location
        end_location = recipient_location or ball_location or default_location

        if start_location is None and end_location is not None and pass_distance_px is not None:
            start_location = self._relative_start_from_end(end_location, team_id, pass_distance_px)
        if end_location is None and start_location is not None and pass_distance_px is not None:
            end_location = self._relative_end_from_start(start_location, team_id, pass_distance_px)

        return start_location or default_location, end_location, recipient_id

    def _infer_carry_locations(
        self,
        player_id: Optional[int],
        team_id: int,
        data: Dict[str, Any],
        freeze_frame: Optional[Dict[str, Any]],
        default_location: Optional[List[float]],
    ) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[int]]:
        details = str(data.get("details") or "")
        parsed = _CARRY_DETAILS_RE.search(details)
        carry_distance_px = float(parsed.group("distance")) if parsed else None

        end_location = self._freeze_frame_player_location(freeze_frame, player_id) or default_location
        start_location = self._last_player_locations.get(player_id)

        if start_location is None and end_location is not None and carry_distance_px is not None:
            start_location = self._relative_start_from_end(end_location, team_id, carry_distance_px)

        return start_location or default_location, end_location, None

    def _relative_start_from_end(
        self,
        end_location: List[float],
        team_id: int,
        distance_px: float,
    ) -> List[float]:
        distance_pitch = (float(distance_px) / float(self.frame_width)) * 120.0
        if team_id == 1:
            start_x = min(120.0, end_location[0] + distance_pitch)
        else:
            start_x = max(0.0, end_location[0] - distance_pitch)
        return [round(start_x, 1), round(float(end_location[1]), 1)]

    def _relative_end_from_start(
        self,
        start_location: List[float],
        team_id: int,
        distance_px: float,
    ) -> List[float]:
        distance_pitch = (float(distance_px) / float(self.frame_width)) * 120.0
        if team_id == 1:
            end_x = max(0.0, start_location[0] - distance_pitch)
        else:
            end_x = min(120.0, start_location[0] + distance_pitch)
        return [round(end_x, 1), round(float(start_location[1]), 1)]

    def _freeze_frame_player_location(
        self,
        freeze_frame: Optional[Dict[str, Any]],
        player_id: Optional[int],
    ) -> Optional[List[float]]:
        if not freeze_frame or player_id is None:
            return None
        for player in freeze_frame.get("players", []):
            if self._safe_int(player.get("player_id"), None) != player_id:
                continue
            return self._convert_coordinates(player.get("location"))
        return None

    def _freeze_frame_ball_location(self, freeze_frame: Optional[Dict[str, Any]]) -> Optional[List[float]]:
        if not freeze_frame:
            return None
        return self._convert_coordinates(freeze_frame.get("ball_location"))

    def _remember_player_location(
        self,
        player_id: Optional[int],
        current_location: Optional[List[float]],
        fallback_location: Optional[List[float]],
    ) -> None:
        if player_id is None:
            return
        location = current_location or fallback_location
        if location is None:
            return
        self._last_player_locations[player_id] = [round(float(location[0]), 1), round(float(location[1]), 1)]

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

    def _create_pass_details(
        self,
        start_location: Optional[List[float]],
        end_location: Optional[List[float]],
        recipient_id: Optional[int],
    ) -> Dict[str, Any]:
        details = {
            "length": 0.0,
            "angle": 0.0,
            "height": {"id": 1, "name": "Ground Pass"},
            "type": {"id": 65, "name": "Open Play"},
        }
        if start_location and end_location:
            dx = float(end_location[0]) - float(start_location[0])
            dy = float(end_location[1]) - float(start_location[1])
            details["length"] = round(math.sqrt(dx * dx + dy * dy), 2)
            details["angle"] = round(math.atan2(dy, dx), 4)
        if end_location:
            details["end_location"] = end_location
        if recipient_id is not None:
            details["recipient"] = self._get_player_info(recipient_id)
        return details

    def _create_carry_details(self, end_location: Optional[List[float]]) -> Dict[str, Any]:
        details: Dict[str, Any] = {}
        if end_location:
            details["end_location"] = end_location
        return details

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
