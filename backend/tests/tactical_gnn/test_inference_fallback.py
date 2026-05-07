from pathlib import Path

from app.tactical_gnn.inference import predict_tactical_snapshot
from app.tactical_gnn.schemas import TacticalGNNConfig


def _event():
    return {"location": [60, 40]}


def _freeze_frame():
    return {
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
        ]
    }


def _heuristic_fallback(event_data, freeze_frame_data):
    return {
        "formation_approx": "Back Four Approx",
        "team_shape": "Compact Shape",
        "attacking_structure": "Balanced Structure",
        "defensive_block": "Mid Block",
        "defensive_shape": "Compact Balanced Mid Block",
        "support_context": "Support is available.",
        "opposition_effect": "The opposition are set.",
    }


def test_predict_tactical_snapshot_falls_back_when_model_missing(tmp_path):
    config = TacticalGNNConfig(model_path=str(tmp_path / "missing.pt"))

    result = predict_tactical_snapshot(
        _event(),
        _freeze_frame(),
        config=config,
        heuristic_fallback=_heuristic_fallback,
    )

    assert result["model_used"] == "heuristic"
    assert result["formation"] == "Back Four Approx"
    assert "not found" in result["graph_metadata"]["fallback_reason"]


def test_predict_tactical_snapshot_falls_back_on_broken_checkpoint(tmp_path):
    checkpoint = tmp_path / "broken.pt"
    checkpoint.write_text("not-a-checkpoint", encoding="utf-8")
    config = TacticalGNNConfig(model_path=str(checkpoint))

    result = predict_tactical_snapshot(
        _event(),
        _freeze_frame(),
        config=config,
        heuristic_fallback=_heuristic_fallback,
    )

    assert result["model_used"] == "heuristic"
    assert result["support_context"] == "Support is available."
    assert result["graph_metadata"]["fallback_reason"]


def test_predict_tactical_snapshot_falls_back_when_freeze_frame_is_insufficient(tmp_path):
    config = TacticalGNNConfig(model_path=str(tmp_path / "unused.pt"), min_players=6)
    result = predict_tactical_snapshot(
        _event(),
        {"freeze_frame": [{"location": [20, 20], "teammate": True}]},
        config=config,
        heuristic_fallback=_heuristic_fallback,
    )

    assert result["model_used"] == "heuristic"
    assert result["graph_metadata"]["insufficient_data"] is True
