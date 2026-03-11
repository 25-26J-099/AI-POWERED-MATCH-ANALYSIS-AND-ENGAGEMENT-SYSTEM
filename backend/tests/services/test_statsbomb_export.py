"""Regression tests for StatsBomb export movement backfilling."""

import json

from app.analytics.player_stats import compute_player_stats
from app.event_detection.statsbomb_export import StatsBombExporter


def test_statsbomb_export_backfills_pass_and_carry_locations_for_xt(tmp_path):
    exporter = StatsBombExporter(frame_width=1280, frame_height=720)

    internal_events = [
        {
            "type": "ball_receipt",
            "frame": 10,
            "timestamp": 0.4,
            "position": [213.3, 360.0],
            "player_id": 1,
            "team_id": 0,
            "details": "Ball receipt",
            "source": "rule",
            "freeze_frame": {
                "event_frame": 10,
                "players": [
                    {"player_id": 1, "location": [213.3, 360.0], "teammate": True, "actor": True, "keeper": False, "team_id": 0},
                    {"player_id": 2, "location": [853.3, 360.0], "teammate": True, "actor": False, "keeper": False, "team_id": 0},
                ],
                "ball_location": [213.3, 360.0],
            },
        },
        {
            "type": "carry",
            "frame": 20,
            "timestamp": 0.8,
            "position": [426.7, 360.0],
            "player_id": 1,
            "team_id": 0,
            "details": "Carry (dist=213px)",
            "source": "rule",
            "freeze_frame": {
                "event_frame": 20,
                "players": [
                    {"player_id": 1, "location": [426.7, 360.0], "teammate": True, "actor": True, "keeper": False, "team_id": 0},
                    {"player_id": 2, "location": [853.3, 360.0], "teammate": True, "actor": False, "keeper": False, "team_id": 0},
                ],
                "ball_location": [426.7, 360.0],
            },
        },
        {
            "type": "pass",
            "frame": 30,
            "timestamp": 1.2,
            "position": [853.3, 360.0],
            "player_id": 1,
            "team_id": 0,
            "details": "Pass #1->2 (d=427px)",
            "source": "ml",
            "freeze_frame": {
                "event_frame": 30,
                "players": [
                    {"player_id": 1, "location": [426.7, 360.0], "teammate": True, "actor": True, "keeper": False, "team_id": 0},
                    {"player_id": 2, "location": [853.3, 360.0], "teammate": True, "actor": False, "keeper": False, "team_id": 0},
                ],
                "ball_location": [853.3, 360.0],
            },
        },
    ]

    output_path = tmp_path / "statsbomb_events.json"
    exporter.export_to_file(internal_events, str(output_path))

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    exported_events = payload["events"]
    carry_event = next(event for event in exported_events if event["type"]["name"] == "Carry")
    pass_event = next(event for event in exported_events if event["type"]["name"] == "Pass")

    assert carry_event["location"] == [20.0, 40.0]
    assert carry_event["carry"]["end_location"] == [40.0, 40.0]
    assert pass_event["location"] == [40.0, 40.0]
    assert pass_event["pass"]["end_location"] == [80.0, 40.0]

    stats = compute_player_stats([carry_event, pass_event])
    assert stats["xt"] > 0.0

