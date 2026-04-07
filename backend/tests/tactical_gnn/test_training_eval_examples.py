import json

from app.tactical_gnn.evaluate import evaluate_checkpoint
from app.tactical_gnn.examples import generate_examples
from app.tactical_gnn.training import train


def _make_dataset(tmp_path):
    payload = {
        "events": [
            {
                "event_uuid": "evt-a",
                "type": "Pass",
                "team_id": 1,
                "player_id": 1,
                "position": [50.0, 40.0],
                "labels": {
                    "formation": "4-3-3",
                    "team_shape": "Wide Shape",
                    "attacking_structure": "Wide Structure",
                    "defensive_block": "Mid Block",
                    "defensive_shape": "Compact Balanced",
                },
                "freeze_frame": {"players": _players(actor_team=1)},
            },
            {
                "event_uuid": "evt-b",
                "type": "Carry",
                "team_id": 2,
                "player_id": 20,
                "position": [40.0, 34.0],
                "labels": {
                    "formation": "4-4-2",
                    "team_shape": "Compact Shape",
                    "attacking_structure": "Central Overload",
                    "defensive_block": "Low Block",
                    "defensive_shape": "Compact Narrow",
                },
                "freeze_frame": {"players": _players(actor_team=2)},
            },
            {
                "event_uuid": "evt-c",
                "type": "Pass",
                "team_id": 1,
                "player_id": 2,
                "position": [54.0, 38.0],
                "labels": {
                    "formation": "4-3-3",
                    "team_shape": "Wide Shape",
                    "attacking_structure": "Wide Structure",
                    "defensive_block": "Mid Block",
                    "defensive_shape": "Compact Balanced",
                },
                "freeze_frame": {"players": _players(actor_team=1, offset=2.0)},
            },
            {
                "event_uuid": "evt-d",
                "type": "Carry",
                "team_id": 2,
                "player_id": 21,
                "position": [42.0, 36.0],
                "labels": {
                    "formation": "4-4-2",
                    "team_shape": "Compact Shape",
                    "attacking_structure": "Central Overload",
                    "defensive_block": "Low Block",
                    "defensive_shape": "Compact Narrow",
                },
                "freeze_frame": {"players": _players(actor_team=2, offset=1.5)},
            },
        ]
    }
    data_path = tmp_path / "events.json"
    data_path.write_text(json.dumps(payload), encoding="utf-8")
    return data_path


def _players(actor_team, offset=0.0):
    team_a = actor_team
    team_b = 1 if actor_team == 2 else 2
    return [
        {"player_id": 1, "team_id": team_a, "location": [12.0 + offset, 10.0], "teammate": True, "actor": True, "keeper": True},
        {"player_id": 2, "team_id": team_a, "location": [20.0 + offset, 20.0], "teammate": True},
        {"player_id": 3, "team_id": team_a, "location": [28.0 + offset, 30.0], "teammate": True},
        {"player_id": 4, "team_id": team_a, "location": [36.0 + offset, 40.0], "teammate": True},
        {"player_id": 5, "team_id": team_a, "location": [44.0 + offset, 50.0], "teammate": True},
        {"player_id": 6, "team_id": team_a, "location": [52.0 + offset, 60.0], "teammate": True},
        {"player_id": 7, "team_id": team_b, "location": [72.0 + offset, 18.0], "teammate": False},
        {"player_id": 8, "team_id": team_b, "location": [80.0 + offset, 28.0], "teammate": False},
        {"player_id": 9, "team_id": team_b, "location": [88.0 + offset, 38.0], "teammate": False},
        {"player_id": 10, "team_id": team_b, "location": [96.0 + offset, 48.0], "teammate": False},
        {"player_id": 11, "team_id": team_b, "location": [102.0 + offset, 58.0], "teammate": False},
        {"player_id": 12, "team_id": team_b, "location": [108.0 + offset, 40.0], "teammate": False, "keeper": True},
    ]


def _make_normalized_jsonl_dataset(tmp_path):
    samples = [
        {
            "event_id": "jsonl-a",
            "event_type": "Pass",
            "event_location": [52.0, 38.0],
            "attacking_right": True,
            "freeze_frame": _players(actor_team=1),
            "labels": {
                "formation": "4-3-3",
                "team_shape": "wide",
                "attacking_structure": "wide_structure",
                "defensive_block": "mid_block",
                "defensive_shape": "compact_balanced",
            },
            "label_source": {head: "synthetic_expert_rule" for head in ["formation", "team_shape", "attacking_structure", "defensive_block", "defensive_shape"]},
        },
        {
            "event_id": "jsonl-b",
            "event_type": "Carry",
            "event_location": [44.0, 34.0],
            "attacking_right": False,
            "freeze_frame": _players(actor_team=2, offset=1.5),
            "labels": {
                "formation": "4-4-2",
                "team_shape": "compact",
                "attacking_structure": "central_overload",
                "defensive_block": "low_block",
                "defensive_shape": "back_five_compact",
            },
            "label_source": {head: "synthetic_expert_rule" for head in ["formation", "team_shape", "attacking_structure", "defensive_block", "defensive_shape"]},
        },
        {
            "event_id": "jsonl-c",
            "event_type": "Pass",
            "event_location": [56.0, 40.0],
            "attacking_right": True,
            "freeze_frame": _players(actor_team=1, offset=2.0),
            "labels": {
                "formation": "4-3-3",
                "team_shape": "vertical",
                "attacking_structure": "vertical_support",
                "defensive_block": "high_press",
                "defensive_shape": "disorganized",
            },
            "label_source": {head: "synthetic_expert_rule" for head in ["formation", "team_shape", "attacking_structure", "defensive_block", "defensive_shape"]},
        },
        {
            "event_id": "jsonl-d",
            "event_type": "Carry",
            "event_location": [40.0, 32.0],
            "attacking_right": False,
            "freeze_frame": _players(actor_team=2, offset=3.0),
            "labels": {
                "formation": "4-4-2",
                "team_shape": "balanced",
                "attacking_structure": "rest_defense_stable",
                "defensive_block": "low_block",
                "defensive_shape": "spread_wide",
            },
            "label_source": {head: "synthetic_expert_rule" for head in ["formation", "team_shape", "attacking_structure", "defensive_block", "defensive_shape"]},
        },
    ]
    path = tmp_path / "gnn_synthetic_augmented.jsonl"
    path.write_text("\n".join(json.dumps(sample) for sample in samples) + "\n", encoding="utf-8")
    return path


def test_training_smoke_on_repo_export_schema(tmp_path):
    data_path = _make_dataset(tmp_path)
    output_dir = tmp_path / "checkpoints"

    result = train(str(data_path), str(output_dir), allow_pseudo_labels=False, patience=1)

    assert (output_dir / "model.pt").exists()
    assert (output_dir / "training_summary.json").exists()
    assert result["usable_samples"] == 4
    assert "formation" in result["active_heads"]


def test_evaluation_and_examples_smoke(tmp_path):
    data_path = _make_dataset(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    train(str(data_path), str(checkpoint_dir), allow_pseudo_labels=False, patience=1)

    eval_dir = tmp_path / "eval"
    metrics = evaluate_checkpoint(str(data_path), str(checkpoint_dir / "model.pt"), str(eval_dir), allow_pseudo_labels=False)
    assert (eval_dir / "metrics.json").exists()
    assert "formation" in metrics["heads"]
    assert metrics["evaluation_split"] == "all"

    examples_dir = tmp_path / "examples"
    generate_examples(str(data_path), str(examples_dir), checkpoint_path=str(checkpoint_dir / "model.pt"), allow_pseudo_labels=False)
    assert (examples_dir / "examples.json").exists()
    assert (examples_dir / "examples.md").exists()


def test_training_smoke_on_normalized_jsonl_schema(tmp_path):
    data_path = _make_normalized_jsonl_dataset(tmp_path)
    output_dir = tmp_path / "normalized-checkpoints"

    result = train(str(data_path), str(output_dir), allow_pseudo_labels=False, patience=1)

    assert (output_dir / "model.pt").exists()
    assert "formation" in result["active_heads"]

    label_maps = json.loads((output_dir / "label_maps.json").read_text(encoding="utf-8"))
    assert "Compact Shape" in label_maps["team_shape"]
    assert "Balanced Structure" in label_maps["attacking_structure"]
    assert "Unknown" in label_maps["defensive_shape"]


def test_evaluation_can_use_saved_validation_split(tmp_path):
    data_path = _make_dataset(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    train(str(data_path), str(checkpoint_dir), allow_pseudo_labels=False, patience=1)

    eval_dir = tmp_path / "eval-val"
    metrics = evaluate_checkpoint(
        str(data_path),
        str(checkpoint_dir / "model.pt"),
        str(eval_dir),
        allow_pseudo_labels=False,
        split="val",
    )

    manifest = json.loads((checkpoint_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert metrics["evaluation_split"] == "val"
    assert metrics["evaluated_samples"] == len(manifest["val_event_ids"])
