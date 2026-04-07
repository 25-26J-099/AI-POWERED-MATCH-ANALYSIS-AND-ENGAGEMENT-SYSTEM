import torch

import app.commentary.tac_commentary as tac
from app.tactical_gnn.model import create_model
from app.tactical_gnn.schemas import TacticalGNNConfig


def _sample_event():
    return {
        "id": "evt-1",
        "type": {"name": "Pass"},
        "player": {"name": "Playmaker"},
        "team": {"name": "Blue FC"},
        "location": [60, 40],
    }


def _sample_events():
    return [_sample_event()]


def _sample_threesixty():
    return {
        "evt-1": {
            "event_uuid": "evt-1",
            "freeze_frame": [
                {"location": [10, 10], "teammate": True, "keeper": True},
                {"location": [20, 20], "teammate": True},
                {"location": [30, 30], "teammate": True},
                {"location": [40, 40], "teammate": True, "actor": True},
                {"location": [50, 50], "teammate": True},
                {"location": [60, 60], "teammate": True},
                {"location": [70, 20], "teammate": False, "keeper": True},
                {"location": [80, 25], "teammate": False},
                {"location": [85, 35], "teammate": False},
                {"location": [90, 45], "teammate": False},
                {"location": [95, 55], "teammate": False},
                {"location": [100, 65], "teammate": False},
            ],
        }
    }


def test_process_event_keeps_commentary_pipeline_working_with_heuristic_fallback():
    result = tac.process_event(_sample_event(), _sample_events(), _sample_threesixty())

    assert result["tactical_description"]
    assert "formation_approx" in result["tactical_labels"]
    assert "support_context" in result["tactical_labels"]


def test_process_event_accepts_gnn_predictions(monkeypatch):
    def fake_predict(event_data, freeze_frame_data, model_path=None, config=None, heuristic_fallback=None):
        return {
            "model_used": "gnn",
            "formation": "4-3-3",
            "formation_confidence": 0.72,
            "team_shape": "Wide Shape",
            "team_shape_confidence": 0.8,
            "attacking_structure": "Wide Structure",
            "attacking_structure_confidence": 0.77,
            "defensive_block": "Mid Block",
            "defensive_block_confidence": 0.69,
            "defensive_shape": "Compact Balanced Mid Block",
            "defensive_shape_confidence": 0.7,
            "support_context": "Support is available.",
            "opposition_effect": "The opposition are set.",
            "graph_metadata": {
                "num_nodes": 12,
                "num_edges": 40,
                "normalization_applied": True,
                "missing_features": [],
            },
        }

    monkeypatch.setattr(tac, "predict_tactical_snapshot", fake_predict)

    result = tac.process_event(_sample_event(), _sample_events(), _sample_threesixty())

    assert result["tactical_labels"]["model_used"] == "gnn"
    assert result["tactical_labels"]["formation_approx"] == "4-3-3"
    assert "4-3-3" in result["tactical_description"]


def test_compose_tactical_description_handles_unclear_unknown_cleanly():
    description = tac.compose_tactical_description(
        "Blue FC",
        "Playmaker",
        "Event type: Pass",
        {
            "formation_approx": "Unclear",
            "team_shape": "Unknown",
            "attacking_structure": "Unknown",
            "defensive_shape": "Unknown",
        },
        "The opposition screen the middle.",
        "Support is nearby.",
    )

    assert "a unclear" not in description.lower()
    assert "a unknown" not in description.lower()
    assert "without a clearly defined base shape" in description.lower()
    assert "without a clearly defined attacking pattern" in description.lower()
    assert "an unclear overall shape" in description.lower()


def test_compose_tactical_description_reduces_structure_repetition():
    description = tac.compose_tactical_description(
        "Blue FC",
        "Playmaker",
        "Event type: Pass",
        {
            "formation_approx": "Unclear",
            "team_shape": "Balanced Shape",
            "attacking_structure": "Balanced Structure",
            "defensive_shape": "Compact Balanced Mid Block",
        },
        "The opposition screen the middle.",
        "Support is nearby.",
    )

    lowered = description.lower()
    assert "balanced structure and a balanced shape" not in lowered
    assert "through a balanced attacking pattern" in lowered


def test_build_spatial_commentary_fallback_varies_by_level():
    tactical_labels = {
        "formation_approx": "4-3-3",
        "team_shape": "Balanced Shape",
        "attacking_structure": "Balanced Structure",
        "defensive_shape": "Compact Balanced Mid Block",
    }

    beginner = tac.build_spatial_commentary_fallback("Beginner", tactical_labels, team_name="Blue FC")
    intermediate = tac.build_spatial_commentary_fallback("Intermediate", tactical_labels, team_name="Blue FC")
    expert = tac.build_spatial_commentary_fallback("Expert", tactical_labels, team_name="Blue FC")

    assert beginner != intermediate
    assert intermediate != expert
    assert "balanced attacking pattern" in expert.lower()


def test_predict_tactical_snapshot_with_real_checkpoint(tmp_path):
    config = TacticalGNNConfig(model_path=str(tmp_path / "tactical_gnn_best.pt"), min_players=4, hidden_dim=32, num_layers=2)
    graph = tac.get_tactical_analysis(
        {"location": [60, 40]},
        _sample_threesixty()["evt-1"],
        prefer_gnn=False,
    )
    assert graph["model_used"] == "heuristic"

    model_graph = tac.get_tactical_analysis(
        {"location": [60, 40]},
        _sample_threesixty()["evt-1"],
        prefer_gnn=False,
    )
    assert model_graph["formation_approx"]

    from app.tactical_gnn.graph_builder import build_graph_from_snapshot

    snapshot_graph = build_graph_from_snapshot({"location": [60, 40]}, _sample_threesixty()["evt-1"], config=config)
    model = create_model(config, snapshot_graph.x.shape[-1], snapshot_graph.edge_attr.shape[-1] if snapshot_graph.edge_attr.numel() else 4)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
                "label_maps": config.label_maps,
                "confidence_threshold": config.confidence_threshold,
            },
            "label_maps": config.label_maps,
            "input_dim": int(snapshot_graph.x.shape[-1]),
            "edge_dim": int(snapshot_graph.edge_attr.shape[-1] if snapshot_graph.edge_attr.numel() else 4),
        },
        config.model_path,
    )

    prediction = tac.get_tactical_analysis(
        {"location": [60, 40]},
        _sample_threesixty()["evt-1"],
        prefer_gnn=True,
        model_path=config.model_path,
        config=config,
    )

    assert prediction["model_used"] == "gnn"
    assert prediction["formation"]
