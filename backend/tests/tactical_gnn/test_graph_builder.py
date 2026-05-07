import torch

from app.tactical_gnn.graph_builder import build_graph_from_snapshot
from app.tactical_gnn.schemas import TacticalGNNConfig


def _freeze_frame(players):
    return {"freeze_frame": players}


def test_graph_builder_handles_malformed_players():
    config = TacticalGNNConfig(min_players=3, k_neighbors=2)
    graph = build_graph_from_snapshot(
        {"location": [50, 40]},
        _freeze_frame(
            [
                {"location": [10, 10], "teammate": True},
                {"location": None, "teammate": True},
                {"location": [30, 30], "teammate": True},
                {"location": "bad", "teammate": False},
                {"location": [80, 20], "teammate": False},
            ]
        ),
        config=config,
    )

    assert graph.metadata.num_nodes == 3
    assert "player_1_location" in graph.metadata.missing_features
    assert "player_3_location" in graph.metadata.missing_features
    assert graph.metadata.insufficient_data is False


def test_graph_builder_supports_variable_player_counts():
    config = TacticalGNNConfig(min_players=4, k_neighbors=2)
    players = [
        {"location": [10, 10], "teammate": True, "keeper": True},
        {"location": [20, 20], "teammate": True},
        {"location": [30, 40], "teammate": True},
        {"location": [40, 50], "teammate": True, "actor": True},
        {"location": [75, 25], "teammate": False, "keeper": True},
        {"location": [85, 35], "teammate": False},
        {"location": [95, 45], "teammate": False},
        {"location": [100, 55], "teammate": False},
    ]

    graph = build_graph_from_snapshot({"location": [42, 44]}, _freeze_frame(players), config=config)

    assert graph.x.shape == (8, 11)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[1] == 4
    assert graph.metadata.num_edges > 0


def test_graph_builder_normalizes_direction_left_to_right():
    config = TacticalGNNConfig(min_players=4, k_neighbors=1)
    players = [
        {"location": [100, 40], "teammate": True, "keeper": True},
        {"location": [90, 30], "teammate": True},
        {"location": [88, 50], "teammate": True},
        {"location": [80, 60], "teammate": True},
        {"location": [20, 40], "teammate": False, "keeper": True},
        {"location": [30, 30], "teammate": False},
        {"location": [35, 50], "teammate": False},
        {"location": [40, 60], "teammate": False},
    ]

    graph = build_graph_from_snapshot({"location": [92, 42]}, _freeze_frame(players), config=config)

    assert graph.metadata.attacking_right is False
    assert torch.isclose(graph.x[0, 0], torch.tensor(20.0 / 120.0), atol=1e-5)


def test_graph_builder_marks_insufficient_snapshots():
    config = TacticalGNNConfig(min_players=6)
    graph = build_graph_from_snapshot(
        {"location": [40, 30]},
        _freeze_frame(
            [
                {"location": [10, 10], "teammate": True},
                {"location": [20, 20], "teammate": True},
                {"location": [30, 30], "teammate": False},
            ]
        ),
        config=config,
    )

    assert graph.metadata.insufficient_data is True
    assert graph.metadata.num_edges == 0
