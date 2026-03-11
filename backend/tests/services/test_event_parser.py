"""Tests for the event parser — especially temporary identity generation."""

from app.services.event_parser import parse_events


def test_events_with_real_names_preserved():
    """Events with player/team names should keep them as-is."""
    raw = [
        {
            "id": "evt-1",
            "type": "Pass",
            "player": {"id": 10, "name": "Lionel Messi"},
            "team": {"id": 1, "name": "Barcelona"},
            "location": [30.5, 40.2],
        }
    ]
    events, players, teams = parse_events(raw, match_id=1)
    assert len(events) == 1
    assert len(players) == 1
    assert len(teams) == 1
    assert players[0].name == "Lionel Messi"
    assert teams[0].name == "Barcelona"


def test_events_with_only_numeric_ids_generate_temp_names():
    """Events with only numeric IDs (no names) should get temporary identities."""
    raw = [
        {
            "id": "evt-2",
            "type": "Pass",
            "player": {"id": 5},
            "team": {"id": 2},
            "location": [50, 50],
        }
    ]
    events, players, teams = parse_events(raw, match_id=1)
    assert len(players) == 1
    assert len(teams) == 1
    assert teams[0].name == "Team 2"
    assert "Player 5" in players[0].name
    # Team context should be included in player name
    assert "Team 2" in players[0].name


def test_events_with_empty_names_generate_temp_names():
    """Events with empty string names should fall back to temporary identities."""
    raw = [
        {
            "id": "evt-3",
            "type": "Shot",
            "player": {"id": 7, "name": ""},
            "team": {"id": 1, "name": ""},
            "location": [100, 40],
        }
    ]
    events, players, teams = parse_events(raw, match_id=1)
    assert len(players) == 1
    assert len(teams) == 1
    assert teams[0].name == "Team 1"
    assert "Player 7" in players[0].name


def test_mixed_events_real_and_numeric():
    """Mix of real names and numeric IDs in same batch."""
    raw = [
        {
            "id": "evt-a",
            "type": "Pass",
            "player": {"id": 1, "name": "Pedri"},
            "team": {"id": 1, "name": "Spain"},
            "location": [40, 30],
        },
        {
            "id": "evt-b",
            "type": "Carry",
            "player": {"id": 99},
            "team": {"id": 2},
            "location": [60, 50],
        },
    ]
    events, players, teams = parse_events(raw, match_id=1)
    assert len(events) == 2
    player_names = {p.name for p in players}
    assert "Pedri" in player_names
    team_names = {t.name for t in teams}
    assert "Spain" in team_names
    assert "Team 2" in team_names


def test_multiple_events_same_player_deduplication():
    """Multiple events for same player/team should not duplicate records."""
    raw = [
        {
            "id": "evt-x",
            "type": "Pass",
            "player": {"id": 3},
            "team": {"id": 1},
            "location": [30, 30],
        },
        {
            "id": "evt-y",
            "type": "Carry",
            "player": {"id": 3},
            "team": {"id": 1},
            "location": [40, 40],
        },
    ]
    events, players, teams = parse_events(raw, match_id=1)
    assert len(events) == 2
    assert len(players) == 1
    assert len(teams) == 1


def test_external_ids_do_not_override_database_primary_keys():
    """Raw provider IDs should not be written into ORM primary key fields."""
    raw = [
        {
            "id": "evt-pk",
            "type": "Pass",
            "player": {"id": 27, "name": "Player 27"},
            "team": {"id": 1, "name": "Team B"},
            "location": [25, 25],
        }
    ]
    events, players, teams = parse_events(raw, match_id=1)
    assert len(events) == 1
    assert len(players) == 1
    assert len(teams) == 1
    assert players[0].id is None
    assert teams[0].id is None
    assert events[0].player_id is None
    assert events[0].team_id is None
