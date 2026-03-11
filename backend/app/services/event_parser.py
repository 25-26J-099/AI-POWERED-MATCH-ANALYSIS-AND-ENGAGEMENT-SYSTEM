"""Parse unified event JSON (events + embedded freeze frames) into ORM objects."""

import uuid
from typing import List, Tuple, Dict
from app.models.models import Event, Player, Team


def _normalize_external_id(value):
    """Treat non-positive or empty ids as missing placeholders."""
    if value in (None, "", 0, "0"):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def parse_events(
    raw_events: List[dict],
    match_id: int,
    existing_players: Dict[str, int] = None,
    existing_teams: Dict[str, int] = None,
) -> Tuple[List[Event], List[Player], List[Team]]:
    """Parse raw event JSON into ORM Event objects.

    The unified format has freeze frames embedded in each event:
    {
        "id": "event_id",
        "type": "Pass",
        "timestamp": "00:00:35",
        "player": {...},
        "location": [33, 48],
        "freeze_frame": [{...}]
    }

    Returns:
        Tuple of (events, new_players, new_teams)
    """
    if existing_players is None:
        existing_players = {}
    if existing_teams is None:
        existing_teams = {}

    events: List[Event] = []
    new_players: List[Player] = []
    new_teams: List[Team] = []

    for raw in raw_events:
        # Event type
        event_type_raw = raw.get("type", "")
        if isinstance(event_type_raw, dict):
            event_type = event_type_raw.get("name", "Unknown")
        else:
            event_type = str(event_type_raw)

        # Player
        player_raw = raw.get("player", {})
        player_id = None
        if player_raw:
            player_name = player_raw.get("name", "")
            pid = _normalize_external_id(player_raw.get("id"))
            if player_name and player_name not in existing_players:
                # Create new player
                player = Player(name=player_name, position=raw.get("position", {}).get("name") if isinstance(raw.get("position"), dict) else None)
                if pid is not None:
                    player.id = pid
                new_players.append(player)
                existing_players[player_name] = pid
            player_id = existing_players.get(player_name)

        # Team
        team_raw = raw.get("team", {})
        team_id = None
        if team_raw:
            team_name = team_raw.get("name", "")
            tid = _normalize_external_id(team_raw.get("id"))
            if team_name and team_name not in existing_teams:
                team = Team(name=team_name)
                if tid is not None:
                    team.id = tid
                new_teams.append(team)
                existing_teams[team_name] = tid
            team_id = existing_teams.get(team_name)

        # Location
        location = raw.get("location", [])
        x = float(location[0]) if len(location) >= 1 else None
        y = float(location[1]) if len(location) >= 2 else None

        # End location (from pass, carry, or shot)
        end_x, end_y = None, None
        for sub_key in ["pass", "carry", "shot"]:
            sub_data = raw.get(sub_key, {})
            if isinstance(sub_data, dict):
                end_loc = sub_data.get("end_location", [])
                if end_loc and len(end_loc) >= 2:
                    end_x = float(end_loc[0])
                    end_y = float(end_loc[1])
                    break

        # Timestamp parsing
        timestamp_raw = raw.get("timestamp", "")
        minute = raw.get("minute", 0)
        second = raw.get("second", 0)

        # If minute/second not directly provided, parse from timestamp
        if minute == 0 and second == 0 and timestamp_raw:
            parts = timestamp_raw.split(":")
            if len(parts) >= 3:
                try:
                    minute = int(parts[1])
                    second = int(float(parts[2]))
                except (ValueError, IndexError):
                    pass

        event_uuid = raw.get("id", str(uuid.uuid4()))

        event = Event(
            match_id=match_id,
            event_uuid=event_uuid,
            event_type=event_type,
            player_id=player_id,
            team_id=team_id,
            period=raw.get("period", 1),
            minute=minute,
            second=second,
            timestamp=timestamp_raw,
            x=x,
            y=y,
            end_x=end_x,
            end_y=end_y,
            raw_data=raw,  # Full JSON including freeze_frame
        )
        events.append(event)

    return events, new_players, new_teams
