import json

from app.tactical_gnn.dataset import prepare_tactical_dataset


def _events_payload():
    return {
        "events": [
            {
                "event_uuid": "evt-1",
                "type": "Pass",
                "team_id": 1,
                "player_id": 10,
                "position": [52.0, 40.0],
                "labels": {
                    "formation": "4-3-3",
                    "team_shape": "Wide Shape",
                    "attacking_structure": "Wide Structure",
                    "defensive_block": "Mid Block",
                    "defensive_shape": "Compact Balanced",
                },
                "freeze_frame": {
                    "players": [
                        {"player_id": 10, "team_id": 1, "location": [12.0, 10.0], "teammate": True, "actor": True},
                        {"player_id": 11, "team_id": 1, "location": [22.0, 18.0], "teammate": True},
                        {"player_id": 12, "team_id": 1, "location": [28.0, 30.0], "teammate": True},
                        {"player_id": 13, "team_id": 1, "location": [36.0, 46.0], "teammate": True},
                        {"player_id": 14, "team_id": 1, "location": [48.0, 60.0], "teammate": True},
                        {"player_id": 15, "team_id": 1, "location": [60.0, 36.0], "teammate": True},
                        {"player_id": 1, "team_id": 2, "location": [72.0, 20.0], "teammate": False},
                        {"player_id": 2, "team_id": 2, "location": [80.0, 28.0], "teammate": False},
                        {"player_id": 3, "team_id": 2, "location": [86.0, 38.0], "teammate": False},
                        {"player_id": 4, "team_id": 2, "location": [92.0, 48.0], "teammate": False},
                        {"player_id": 5, "team_id": 2, "location": [98.0, 58.0], "teammate": False},
                        {"player_id": 6, "team_id": 2, "location": [104.0, 40.0], "teammate": False, "keeper": True},
                    ]
                },
            },
            {
                "event_uuid": "evt-2",
                "type": "Carry",
                "team_id": 2,
                "player_id": 20,
                "position": [45.0, 30.0],
                "freeze_frame": {
                    "players": [
                        {"player_id": 20, "team_id": 2, "location": [16.0, 14.0], "teammate": True, "actor": True},
                        {"player_id": 21, "team_id": 2, "location": [26.0, 24.0], "teammate": True},
                        {"player_id": 22, "team_id": 2, "location": [30.0, 34.0], "teammate": True},
                        {"player_id": 23, "team_id": 2, "location": [34.0, 44.0], "teammate": True},
                        {"player_id": 24, "team_id": 2, "location": [38.0, 54.0], "teammate": True},
                        {"player_id": 25, "team_id": 2, "location": [42.0, 64.0], "teammate": True},
                        {"player_id": 7, "team_id": 1, "location": [74.0, 18.0], "teammate": False},
                        {"player_id": 8, "team_id": 1, "location": [78.0, 28.0], "teammate": False},
                        {"player_id": 9, "team_id": 1, "location": [82.0, 38.0], "teammate": False},
                        {"player_id": 16, "team_id": 1, "location": [86.0, 48.0], "teammate": False},
                        {"player_id": 17, "team_id": 1, "location": [90.0, 58.0], "teammate": False},
                        {"player_id": 18, "team_id": 1, "location": [104.0, 40.0], "teammate": False, "keeper": True},
                    ]
                },
            },
            {
                "event_uuid": "evt-3",
                "type": "Pass",
                "team_id": 1,
                "player_id": 10,
            },
        ]
    }


def test_prepare_tactical_dataset_detects_repo_event_schema_and_label_provenance(tmp_path):
    dataset_path = tmp_path / "events.json"
    dataset_path.write_text(json.dumps(_events_payload()), encoding="utf-8")

    samples, report = prepare_tactical_dataset(dataset_path, allow_pseudo_labels=True)

    assert report.detected_formats == ["events"]
    assert report.usable_samples == 2
    assert report.dropped_reasons["missing_freeze_frame"] == 1
    assert samples[0]["label_sources"]["formation"] == "ground_truth"
    assert samples[1]["label_sources"]["formation"] == "pseudo_heuristic"


def test_prepare_tactical_dataset_can_disable_pseudo_labels(tmp_path):
    dataset_path = tmp_path / "events.json"
    dataset_path.write_text(json.dumps(_events_payload()), encoding="utf-8")

    samples, _ = prepare_tactical_dataset(dataset_path, allow_pseudo_labels=False)

    assert samples[1]["labels"]["formation"] is None
    assert samples[1]["label_sources"]["formation"] == "missing"


def test_prepare_tactical_dataset_normalizes_synthetic_jsonl_labels(tmp_path):
    dataset_path = tmp_path / "gnn_synthetic_augmented.jsonl"
    sample = {
        "event_id": "synthetic-1",
        "event_type": "Pass",
        "event_location": [54.0, 38.0],
        "attacking_right": True,
        "freeze_frame": [
            {"player_id": 1, "team_id": 1, "location": [12.0, 14.0], "teammate": True, "actor": True},
            {"player_id": 2, "team_id": 1, "location": [24.0, 24.0], "teammate": True},
            {"player_id": 3, "team_id": 1, "location": [30.0, 36.0], "teammate": True},
            {"player_id": 4, "team_id": 1, "location": [42.0, 48.0], "teammate": True},
            {"player_id": 5, "team_id": 1, "location": [52.0, 58.0], "teammate": True},
            {"player_id": 6, "team_id": 1, "location": [64.0, 34.0], "teammate": True},
            {"player_id": 7, "team_id": 2, "location": [72.0, 18.0], "teammate": False},
            {"player_id": 8, "team_id": 2, "location": [80.0, 28.0], "teammate": False},
            {"player_id": 9, "team_id": 2, "location": [88.0, 38.0], "teammate": False},
            {"player_id": 10, "team_id": 2, "location": [96.0, 48.0], "teammate": False},
            {"player_id": 11, "team_id": 2, "location": [104.0, 58.0], "teammate": False},
            {"player_id": 12, "team_id": 2, "location": [108.0, 40.0], "teammate": False, "keeper": True},
        ],
        "labels": {
            "formation": "unknown",
            "team_shape": "compact",
            "attacking_structure": "vertical_support",
            "defensive_block": "mid_block",
            "defensive_shape": "back_five_compact",
        },
        "label_source": {
            "formation": "synthetic_expert_rule",
            "team_shape": "synthetic_expert_rule",
            "attacking_structure": "synthetic_expert_rule",
            "defensive_block": "synthetic_expert_rule",
            "defensive_shape": "synthetic_expert_rule",
        },
    }
    dataset_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    samples, report = prepare_tactical_dataset(dataset_path, allow_pseudo_labels=False)

    assert report.detected_formats == ["generic"]
    assert report.usable_samples == 1
    assert samples[0]["labels"]["formation"] == "Unclear"
    assert samples[0]["labels"]["team_shape"] == "Compact Shape"
    assert samples[0]["labels"]["attacking_structure"] == "Vertical Support Structure"
    assert samples[0]["labels"]["defensive_block"] == "Mid Block"
    assert samples[0]["labels"]["defensive_shape"] == "Compact Balanced"
    assert samples[0]["label_sources"]["formation"] == "synthetic_expert_rule"
    assert samples[0]["metadata"]["raw_labels"]["defensive_shape"] == "back_five_compact"
