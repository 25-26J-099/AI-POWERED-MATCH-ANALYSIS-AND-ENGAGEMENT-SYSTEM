import torch

from app.tactical_gnn.graph_builder import build_graph_from_snapshot
from app.tactical_gnn.model import create_model
from app.tactical_gnn.schemas import TacticalGNNConfig


def test_tactical_gnn_model_forward_shapes():
    config = TacticalGNNConfig(min_players=4, hidden_dim=32, num_layers=2)
    graph = build_graph_from_snapshot(
        {"location": [50, 40]},
        {
            "freeze_frame": [
                {"location": [10, 10], "teammate": True, "keeper": True},
                {"location": [20, 20], "teammate": True},
                {"location": [30, 30], "teammate": True},
                {"location": [40, 40], "teammate": True, "actor": True},
                {"location": [70, 20], "teammate": False, "keeper": True},
                {"location": [80, 30], "teammate": False},
                {"location": [90, 40], "teammate": False},
                {"location": [100, 50], "teammate": False},
            ]
        },
        config=config,
    )
    model = create_model(config, graph.x.shape[-1], graph.edge_attr.shape[-1] if graph.edge_attr.numel() else 4)

    outputs = model(graph)

    assert outputs["embedding"].shape == (1, config.hidden_dim)
    assert outputs["formation"].shape == (1, len(config.label_maps["formation"]))
    assert outputs["team_shape"].shape == (1, len(config.label_maps["team_shape"]))
    assert outputs["attacking_structure"].shape == (1, len(config.label_maps["attacking_structure"]))
    assert outputs["defensive_block"].shape == (1, len(config.label_maps["defensive_block"]))
    assert outputs["defensive_shape"].shape == (1, len(config.label_maps["defensive_shape"]))
